use std::iter;

use super::{
    circuit::{Advice, Assignment, Circuit, Column, ConstraintSystem, Fixed},
    Error, Proof, ProvingKey,
};
use crate::arithmetic::{
    eval_polynomial, get_challenge_scalar, parallelize, BatchInvert, Challenge, Curve, CurveAffine,
    Field,
};
use crate::poly::{
    commitment::{Blind, Params},
    multiopen::{self, ProverQuery},
    LagrangeCoeff, Polynomial, Rotation,
};
use crate::transcript::{Hasher, Transcript};

impl<C: CurveAffine> Proof<C> {
    /// This creates a proof for the provided `circuit` when given the public
    /// parameters `params` and the proving key [`ProvingKey`] that was
    /// generated previously for the same circuit.
    pub fn create<
        HBase: Hasher<C::Base>,
        HScalar: Hasher<C::Scalar>,
        ConcreteCircuit: Circuit<C::Scalar>,
    >(
        params: &Params<C>,
        pk: &ProvingKey<C>,
        circuit: &ConcreteCircuit,
        aux: &[Polynomial<C::Scalar, LagrangeCoeff>],
    ) -> Result<Self, Error> {
        if aux.len() != pk.vk.cs.num_aux_columns {
            return Err(Error::IncompatibleParams);
        }

        struct WitnessCollection<F: Field> {
            advice: Vec<Polynomial<F, LagrangeCoeff>>,
            _marker: std::marker::PhantomData<F>,
        }

        impl<F: Field> Assignment<F> for WitnessCollection<F> {
            fn assign_advice(
                &mut self,
                column: Column<Advice>,
                row: usize,
                to: impl FnOnce() -> Result<F, Error>,
            ) -> Result<(), Error> {
                *self
                    .advice
                    .get_mut(column.index())
                    .and_then(|v| v.get_mut(row))
                    .ok_or(Error::BoundsFailure)? = to()?;

                Ok(())
            }

            fn assign_fixed(
                &mut self,
                _: Column<Fixed>,
                _: usize,
                _: impl FnOnce() -> Result<F, Error>,
            ) -> Result<(), Error> {
                // We only care about advice columns here

                Ok(())
            }

            fn copy(
                &mut self,
                _: usize,
                _: usize,
                _: usize,
                _: usize,
                _: usize,
            ) -> Result<(), Error> {
                // We only care about advice columns here

                Ok(())
            }
        }

        let domain = &pk.vk.domain;
        let mut meta = ConstraintSystem::default();
        let config = ConcreteCircuit::configure(&mut meta);

        let mut witness = WitnessCollection {
            advice: vec![domain.empty_lagrange(); meta.num_advice_columns],
            _marker: std::marker::PhantomData,
        };

        // Synthesize the circuit to obtain the witness and other information.
        circuit.synthesize(&mut witness, config)?;

        let witness = witness;

        // Create a transcript for obtaining Fiat-Shamir challenges.
        let mut transcript = Transcript::<C, HBase, HScalar>::new();

        // Compute commitments to aux column polynomials
        let aux_commitments_projective: Vec<_> = aux
            .iter()
            .map(|poly| params.commit_lagrange(poly, Blind::default()))
            .collect();
        let mut aux_commitments = vec![C::zero(); aux_commitments_projective.len()];
        C::Projective::batch_to_affine(&aux_commitments_projective, &mut aux_commitments);
        let aux_commitments = aux_commitments;
        drop(aux_commitments_projective);
        metrics::counter!("aux_commitments", aux_commitments.len() as u64);

        for commitment in &aux_commitments {
            transcript
                .absorb_point(commitment)
                .map_err(|_| Error::TranscriptError)?;
        }

        let aux_polys: Vec<_> = aux
            .iter()
            .map(|poly| {
                let lagrange_vec = domain.lagrange_from_vec(poly.to_vec());
                domain.lagrange_to_coeff(lagrange_vec)
            })
            .collect();

        let aux_cosets: Vec<_> = meta
            .aux_queries
            .iter()
            .map(|&(column, at)| {
                let poly = aux_polys[column.index()].clone();
                domain.coeff_to_extended(poly, at)
            })
            .collect();

        // Compute commitments to advice column polynomials
        let advice_blinds: Vec<_> = witness
            .advice
            .iter()
            .map(|_| Blind(C::Scalar::random()))
            .collect();
        let advice_commitments_projective: Vec<_> = witness
            .advice
            .iter()
            .zip(advice_blinds.iter())
            .map(|(poly, blind)| params.commit_lagrange(poly, *blind))
            .collect();
        let mut advice_commitments = vec![C::zero(); advice_commitments_projective.len()];
        C::Projective::batch_to_affine(&advice_commitments_projective, &mut advice_commitments);
        let advice_commitments = advice_commitments;
        drop(advice_commitments_projective);
        metrics::counter!("advice_commitments", advice_commitments.len() as u64);

        for commitment in &advice_commitments {
            transcript
                .absorb_point(commitment)
                .map_err(|_| Error::TranscriptError)?;
        }

        let advice_polys: Vec<_> = witness
            .advice
            .clone()
            .into_iter()
            .map(|poly| domain.lagrange_to_coeff(poly))
            .collect();

        let advice_cosets: Vec<_> = meta
            .advice_queries
            .iter()
            .map(|&(column, at)| {
                let poly = advice_polys[column.index()].clone();
                domain.coeff_to_extended(poly, at)
            })
            .collect();

        // Sample x_0 challenge
        let x_0: C::Scalar = get_challenge_scalar(Challenge(transcript.squeeze().get_lower_128()));

        // Sample x_1 challenge
        let x_1: C::Scalar = get_challenge_scalar(Challenge(transcript.squeeze().get_lower_128()));

        // Compute permutation product polynomial commitment
        let mut permutation_product_polys = vec![];
        let mut permutation_product_cosets = vec![];
        let mut permutation_product_cosets_inv = vec![];
        let mut permutation_product_commitments_projective = vec![];
        let mut permutation_product_blinds = vec![];

        // Iterate over each permutation
        let mut permutation_modified_advice = pk
            .vk
            .cs
            .permutations
            .iter()
            .zip(pk.permutations.iter())
            // Goal is to compute the products of fractions
            //
            // (p_j(\omega^i) + \delta^j \omega^i \beta + \gamma) /
            // (p_j(\omega^i) + \beta s_j(\omega^i) + \gamma)
            //
            // where p_j(X) is the jth advice column in this permutation,
            // and i is the ith row of the column.
            .map(|(columns, permuted_values)| {
                let mut modified_advice = vec![C::Scalar::one(); params.n as usize];

                // Iterate over each column of the permutation
                for (&column, permuted_column_values) in columns.iter().zip(permuted_values.iter())
                {
                    parallelize(&mut modified_advice, |modified_advice, start| {
                        for ((modified_advice, advice_value), permuted_advice_value) in
                            modified_advice
                                .iter_mut()
                                .zip(witness.advice[column.index()][start..].iter())
                                .zip(permuted_column_values[start..].iter())
                        {
                            *modified_advice *=
                                &(x_0 * permuted_advice_value + &x_1 + advice_value);
                        }
                    });
                }

                modified_advice
            })
            .collect::<Vec<_>>();

        // Batch invert to obtain the denominators for the permutation product
        // polynomials
        permutation_modified_advice
            .iter_mut()
            .flat_map(|v| v.iter_mut())
            .batch_invert();

        for (columns, mut modified_advice) in pk
            .vk
            .cs
            .permutations
            .iter()
            .zip(permutation_modified_advice.into_iter())
        {
            // Iterate over each column again, this time finishing the computation
            // of the entire fraction by computing the numerators
            let mut deltaomega = C::Scalar::one();
            for &column in columns.iter() {
                let omega = domain.get_omega();
                parallelize(&mut modified_advice, |modified_advice, start| {
                    let mut deltaomega = deltaomega * &omega.pow_vartime(&[start as u64, 0, 0, 0]);
                    for (modified_advice, advice_value) in modified_advice
                        .iter_mut()
                        .zip(witness.advice[column.index()][start..].iter())
                    {
                        // Multiply by p_j(\omega^i) + \delta^j \omega^i \beta
                        *modified_advice *= &(deltaomega * &x_0 + &x_1 + advice_value);
                        deltaomega *= &omega;
                    }
                });
                deltaomega *= &C::Scalar::DELTA;
            }

            // The modified_advice vector is a vector of products of fractions
            // of the form
            //
            // (p_j(\omega^i) + \delta^j \omega^i \beta + \gamma) /
            // (p_j(\omega^i) + \beta s_j(\omega^i) + \gamma)
            //
            // where i is the index into modified_advice, for the jth column in
            // the permutation

            // Compute the evaluations of the permutation product polynomial
            // over our domain, starting with z[0] = 1
            let mut z = vec![C::Scalar::one()];
            for row in 1..(params.n as usize) {
                let mut tmp = z[row - 1];

                tmp *= &modified_advice[row];
                z.push(tmp);
            }
            let z = domain.lagrange_from_vec(z);

            let blind = Blind(C::Scalar::random());

            permutation_product_commitments_projective.push(params.commit_lagrange(&z, blind));
            permutation_product_blinds.push(blind);
            let z = domain.lagrange_to_coeff(z);
            permutation_product_polys.push(z.clone());
            permutation_product_cosets
                .push(domain.coeff_to_extended(z.clone(), Rotation::default()));
            permutation_product_cosets_inv.push(domain.coeff_to_extended(z, Rotation(-1)));
        }
        let mut permutation_product_commitments =
            vec![C::zero(); permutation_product_commitments_projective.len()];
        C::Projective::batch_to_affine(
            &permutation_product_commitments_projective,
            &mut permutation_product_commitments,
        );
        let permutation_product_commitments = permutation_product_commitments;
        drop(permutation_product_commitments_projective);

        // Hash each permutation product commitment
        for c in &permutation_product_commitments {
            transcript
                .absorb_point(c)
                .map_err(|_| Error::TranscriptError)?;
        }

        // Obtain challenge for keeping all separate gates linearly independent
        let x_2: C::Scalar = get_challenge_scalar(Challenge(transcript.squeeze().get_lower_128()));

        // Evaluate the h(X) polynomial's constraint system expressions for the constraints provided
        let h_poly =
            iter::empty()
                // Custom constraints
                .chain(meta.gates.iter().map(|poly| {
                    poly.evaluate(
                        &|index| pk.fixed_cosets[index].clone(),
                        &|index| advice_cosets[index].clone(),
                        &|index| aux_cosets[index].clone(),
                        &|a, b| a + &b,
                        &|a, b| a * &b,
                        &|a, scalar| a * scalar,
                    )
                }))
                // l_0(X) * (1 - z(X)) = 0
                .chain(
                    permutation_product_cosets
                        .iter()
                        .cloned()
                        .map(|coset| Polynomial::one_minus(coset) * &pk.l0),
                )
                // z(X) \prod (p(X) + \beta s_i(X) + \gamma) - z(omega^{-1} X) \prod (p(X) + \delta^i \beta X + \gamma)
                .chain(pk.vk.cs.permutations.iter().enumerate().map(
                    |(permutation_index, columns)| {
                        let mut left = permutation_product_cosets[permutation_index].clone();
                        for (advice, permutation) in columns
                            .iter()
                            .map(|&column| {
                                &advice_cosets[pk.vk.cs.get_advice_query_index(column, 0)]
                            })
                            .zip(pk.permutation_cosets[permutation_index].iter())
                        {
                            parallelize(&mut left, |left, start| {
                                for ((left, advice), permutation) in left
                                    .iter_mut()
                                    .zip(advice[start..].iter())
                                    .zip(permutation[start..].iter())
                                {
                                    *left *= &(*advice + &(x_0 * permutation) + &x_1);
                                }
                            });
                        }

                        let mut right = permutation_product_cosets_inv[permutation_index].clone();
                        let mut current_delta = x_0 * &C::Scalar::ZETA;
                        let step = domain.get_extended_omega();
                        for advice in columns.iter().map(|&column| {
                            &advice_cosets[pk.vk.cs.get_advice_query_index(column, 0)]
                        }) {
                            parallelize(&mut right, move |right, start| {
                                let mut beta_term =
                                    current_delta * &step.pow_vartime(&[start as u64, 0, 0, 0]);
                                for (right, advice) in right.iter_mut().zip(advice[start..].iter())
                                {
                                    *right *= &(*advice + &beta_term + &x_1);
                                    beta_term *= &step;
                                }
                            });
                            current_delta *= &C::Scalar::DELTA;
                        }

                        left - &right
                    },
                ))
                .fold(domain.empty_extended(), |h_poly, v| h_poly * x_2 + &v);

        // Divide by t(X) = X^{params.n} - 1.
        let h_poly = domain.divide_by_vanishing_poly(h_poly);

        // Obtain final h(X) polynomial
        let h_poly = domain.extended_to_coeff(h_poly);

        // Split h(X) up into pieces
        let h_pieces = h_poly
            .chunks_exact(params.n as usize)
            .map(|v| domain.coeff_from_vec(v.to_vec()))
            .collect::<Vec<_>>();
        drop(h_poly);
        let h_blinds: Vec<_> = h_pieces
            .iter()
            .map(|_| Blind(C::Scalar::random()))
            .collect();

        // Compute commitments to each h(X) piece
        let h_commitments_projective: Vec<_> = h_pieces
            .iter()
            .zip(h_blinds.iter())
            .map(|(h_piece, blind)| params.commit(&h_piece, *blind))
            .collect();
        let mut h_commitments = vec![C::zero(); h_commitments_projective.len()];
        C::Projective::batch_to_affine(&h_commitments_projective, &mut h_commitments);
        let h_commitments = h_commitments;
        drop(h_commitments_projective);

        // Hash each h(X) piece
        for c in h_commitments.iter() {
            transcript
                .absorb_point(c)
                .map_err(|_| Error::TranscriptError)?;
        }

        let x_3: C::Scalar = get_challenge_scalar(Challenge(transcript.squeeze().get_lower_128()));
        let x_3_inv = domain.rotate_omega(x_3, Rotation(-1));

        // Evaluate polynomials at omega^i x_3
        let advice_evals: Vec<_> = meta
            .advice_queries
            .iter()
            .map(|&(column, at)| {
                eval_polynomial(&advice_polys[column.index()], domain.rotate_omega(x_3, at))
            })
            .collect();

        let aux_evals: Vec<_> = meta
            .aux_queries
            .iter()
            .map(|&(column, at)| {
                eval_polynomial(&aux_polys[column.index()], domain.rotate_omega(x_3, at))
            })
            .collect();

        let fixed_evals: Vec<_> = meta
            .fixed_queries
            .iter()
            .map(|&(column, at)| {
                eval_polynomial(
                    &pk.fixed_polys[column.index()],
                    domain.rotate_omega(x_3, at),
                )
            })
            .collect();

        let permutation_product_evals: Vec<C::Scalar> = permutation_product_polys
            .iter()
            .map(|poly| eval_polynomial(poly, x_3))
            .collect();

        let permutation_product_inv_evals: Vec<C::Scalar> = permutation_product_polys
            .iter()
            .map(|poly| eval_polynomial(poly, domain.rotate_omega(x_3, Rotation(-1))))
            .collect();

        let permutation_evals: Vec<Vec<C::Scalar>> = pk
            .permutation_polys
            .iter()
            .map(|polys| {
                polys
                    .iter()
                    .map(|poly| eval_polynomial(poly, x_3))
                    .collect()
            })
            .collect();

        let h_evals: Vec<_> = h_pieces
            .iter()
            .map(|poly| eval_polynomial(poly, x_3))
            .collect();

        // Hash each advice evaluation
        for eval in advice_evals
            .iter()
            .chain(aux_evals.iter())
            .chain(fixed_evals.iter())
            .chain(h_evals.iter())
            .chain(permutation_product_evals.iter())
            .chain(permutation_product_inv_evals.iter())
            .chain(permutation_evals.iter().flat_map(|evals| evals.iter()))
        {
            transcript.absorb_scalar(*eval);
        }

        let instances =
            iter::empty()
                .chain(pk.vk.cs.advice_queries.iter().enumerate().map(
                    |(query_index, &(column, at))| ProverQuery {
                        point: domain.rotate_omega(x_3, at),
                        poly: &advice_polys[column.index()],
                        blind: advice_blinds[column.index()],
                        eval: advice_evals[query_index],
                    },
                ))
                .chain(pk.vk.cs.aux_queries.iter().enumerate().map(
                    |(query_index, &(column, at))| ProverQuery {
                        point: domain.rotate_omega(x_3, at),
                        poly: &aux_polys[column.index()],
                        blind: Blind::default(),
                        eval: aux_evals[query_index],
                    },
                ))
                .chain(pk.vk.cs.fixed_queries.iter().enumerate().map(
                    |(query_index, &(column, at))| ProverQuery {
                        point: domain.rotate_omega(x_3, at),
                        poly: &pk.fixed_polys[column.index()],
                        blind: Blind::default(),
                        eval: fixed_evals[query_index],
                    },
                ))
                // We query the h(X) polynomial at x_3
                .chain(
                    h_pieces
                        .iter()
                        .zip(h_blinds.iter())
                        .zip(h_evals.iter())
                        .map(|((h_poly, h_blind), h_eval)| ProverQuery {
                            point: x_3,
                            poly: h_poly,
                            blind: *h_blind,
                            eval: *h_eval,
                        }),
                );

        // Handle permutation arguments, if any exist
        let permutation_instances = if !pk.vk.cs.permutations.is_empty() {
            Some(
                iter::empty()
                    // Open permutation product commitments at x_3
                    .chain(
                        permutation_product_polys
                            .iter()
                            .zip(permutation_product_blinds.iter())
                            .zip(permutation_product_evals.iter())
                            .map(|((poly, blind), eval)| ProverQuery {
                                point: x_3,
                                poly,
                                blind: *blind,
                                eval: *eval,
                            }),
                    )
                    // Open permutation polynomial commitments at x_3
                    .chain(
                        pk.permutation_polys
                            .iter()
                            .zip(permutation_evals.iter())
                            .flat_map(|(polys, evals)| polys.iter().zip(evals.iter()))
                            .map(|(poly, eval)| ProverQuery {
                                point: x_3,
                                poly,
                                blind: Blind::default(),
                                eval: *eval,
                            }),
                    )
                    // Open permutation product commitments at \omega^{-1} x_3
                    .chain(
                        permutation_product_polys
                            .iter()
                            .zip(permutation_product_blinds.iter())
                            .zip(permutation_product_inv_evals.iter())
                            .map(|((poly, blind), eval)| ProverQuery {
                                point: x_3_inv,
                                poly,
                                blind: *blind,
                                eval: *eval,
                            }),
                    ),
            )
        } else {
            None
        };

        let multiopening = multiopen::Proof::create(
            params,
            &mut transcript,
            instances.chain(permutation_instances.into_iter().flatten()),
        )
        .map_err(|_| Error::OpeningError)?;

        Ok(Proof {
            advice_commitments,
            h_commitments,
            permutation_product_commitments,
            permutation_product_evals,
            permutation_product_inv_evals,
            permutation_evals,
            advice_evals,
            fixed_evals,
            aux_evals,
            h_evals,
            multiopening,
        })
    }
}
