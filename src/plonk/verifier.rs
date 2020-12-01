use std::iter;

use super::{
    ChallengeBeta, ChallengeGamma, ChallengeTheta, ChallengeX, ChallengeY, Error, Proof,
    VerifyingKey,
};
use crate::arithmetic::{CurveAffine, Field};
use crate::poly::{
    commitment::{Guard, Params, MSM},
    multiopen::VerifierQuery,
    Rotation,
};
use crate::transcript::{Hasher, Transcript};

impl<'a, C: CurveAffine> Proof<C> {
    /// Returns a boolean indicating whether or not the proof is valid
    pub fn verify<HBase: Hasher<C::Base>, HScalar: Hasher<C::Scalar>>(
        &'a self,
        params: &'a Params<C>,
        vk: &'a VerifyingKey<C>,
        msm: MSM<'a, C>,
        aux_commitments: &'a [C],
    ) -> Result<Guard<'a, C>, Error> {
        self.check_lengths(vk, aux_commitments)?;

        // Check that aux_commitments matches the expected number of aux_columns
        // and self.aux_evals
        if aux_commitments.len() != vk.cs.num_aux_columns
            || self.aux_evals.len() != vk.cs.num_aux_columns
        {
            return Err(Error::IncompatibleParams);
        }

        // Create a transcript for obtaining Fiat-Shamir challenges.
        let mut transcript = Transcript::<C, HBase, HScalar>::new();

        // Hash the aux (external) commitments into the transcript
        for commitment in aux_commitments {
            transcript
                .absorb_point(commitment)
                .map_err(|_| Error::TranscriptError)?;
        }

        // Hash the prover's advice commitments into the transcript
        for commitment in &self.advice_commitments {
            transcript
                .absorb_point(commitment)
                .map_err(|_| Error::TranscriptError)?;
        }

        // Sample theta challenge for keeping lookup columns linearly independent
        let theta = ChallengeTheta::get(&mut transcript);

        for lookup in &self.lookup_proofs {
            transcript
                .absorb_point(&lookup.permuted_input_commitment)
                .map_err(|_| Error::TranscriptError)?;
            transcript
                .absorb_point(&lookup.permuted_table_commitment)
                .map_err(|_| Error::TranscriptError)?;
        }

        // Sample beta challenge
        let beta = ChallengeBeta::get(&mut transcript);

        // Sample gamma challenge
        let gamma = ChallengeGamma::get(&mut transcript);

        // Hash each permutation product commitment
        if let Some(p) = &self.permutations {
            p.absorb_commitments(&mut transcript)?;
        }

        // Hash each lookup product commitment
        for lookup in &self.lookup_proofs {
            transcript
                .absorb_point(&lookup.product_commitment)
                .map_err(|_| Error::TranscriptError)?;
        }

        // Sample y challenge, which keeps the gates linearly independent.
        let y = ChallengeY::get(&mut transcript);

        // Obtain a commitment to h(X) in the form of multiple pieces of degree n - 1
        for c in &self.h_commitments {
            transcript
                .absorb_point(c)
                .map_err(|_| Error::TranscriptError)?;
        }

        // Sample x challenge, which is used to ensure the circuit is
        // satisfied with high probability.
        let x = ChallengeX::get(&mut transcript);

        // This check ensures the circuit is satisfied so long as the polynomial
        // commitments open to the correct values.
        self.check_hx(params, vk, theta, beta, gamma, y, x)?;

        for eval in self
            .advice_evals
            .iter()
            .chain(self.aux_evals.iter())
            .chain(self.fixed_evals.iter())
            .chain(self.h_evals.iter())
            .chain(
                self.lookup_proofs
                    .iter()
                    .flat_map(|proof| {
                        std::iter::empty()
                            .chain(Some(proof.product_eval))
                            .chain(Some(proof.product_inv_eval))
                            .chain(Some(proof.permuted_input_eval))
                            .chain(Some(proof.permuted_input_inv_eval))
                            .chain(Some(proof.permuted_table_eval))
                    })
                    .collect::<Vec<_>>()
                    .iter(),
            )
            .chain(
                self.permutations
                    .as_ref()
                    .map(|p| p.evals())
                    .into_iter()
                    .flatten(),
            )
        {
            transcript.absorb_scalar(*eval);
        }

        let queries =
            iter::empty()
                .chain(vk.cs.advice_queries.iter().enumerate().map(
                    |(query_index, &(column, at))| VerifierQuery {
                        point: vk.domain.rotate_omega(*x, at),
                        commitment: &self.advice_commitments[column.index()],
                        eval: self.advice_evals[query_index],
                    },
                ))
                .chain(
                    vk.cs
                        .aux_queries
                        .iter()
                        .enumerate()
                        .map(|(query_index, &(column, at))| VerifierQuery {
                            point: vk.domain.rotate_omega(*x, at),
                            commitment: &aux_commitments[column.index()],
                            eval: self.aux_evals[query_index],
                        }),
                )
                .chain(vk.cs.fixed_queries.iter().enumerate().map(
                    |(query_index, &(column, at))| VerifierQuery {
                        point: vk.domain.rotate_omega(*x, at),
                        commitment: &vk.fixed_commitments[column.index()],
                        eval: self.fixed_evals[query_index],
                    },
                ))
                .chain(
                    self.h_commitments
                        .iter()
                        .enumerate()
                        .zip(self.h_evals.iter())
                        .map(|((idx, _), &eval)| VerifierQuery {
                            point: *x,
                            commitment: &self.h_commitments[idx],
                            eval,
                        }),
                )
                .chain(
                    // Handle lookup arguments, if any exist
                    self.lookup_proofs.iter().flat_map(|lookup| {
                        vec![
                            // Open lookup product commitments at x
                            VerifierQuery {
                                point: *x,
                                commitment: &lookup.product_commitment,
                                eval: lookup.product_eval,
                            },
                            // Open lookup input commitments at x
                            VerifierQuery {
                                point: *x,
                                commitment: &lookup.permuted_input_commitment,
                                eval: lookup.permuted_input_eval,
                            },
                            // Open lookup table commitments at x
                            VerifierQuery {
                                point: *x,
                                commitment: &lookup.permuted_table_commitment,
                                eval: lookup.permuted_table_eval,
                            },
                            // Open lookup input commitments at \omega^{-1} x
                            VerifierQuery {
                                point: vk.domain.rotate_omega(*x, Rotation(-1)),
                                commitment: &lookup.permuted_input_commitment,
                                eval: lookup.permuted_input_inv_eval,
                            },
                            // Open lookup product commitments at \omega^{-1} x
                            VerifierQuery {
                                point: vk.domain.rotate_omega(*x, Rotation(-1)),
                                commitment: &lookup.product_commitment,
                                eval: lookup.product_inv_eval,
                            },
                        ]
                    }),
                );

        // We are now convinced the circuit is satisfied so long as the
        // polynomial commitments open to the correct values.
        self.multiopening
            .verify(
                params,
                &mut transcript,
                queries.chain(
                    self.permutations
                        .as_ref()
                        .map(|p| p.queries(vk, *x))
                        .into_iter()
                        .flatten(),
                ),
                msm,
            )
            .map_err(|_| Error::OpeningError)
    }

    /// Checks that the lengths of vectors are consistent with the constraint
    /// system
    fn check_lengths(&self, vk: &VerifyingKey<C>, aux_commitments: &[C]) -> Result<(), Error> {
        // Check that aux_commitments matches the expected number of aux_columns
        // and self.aux_evals
        if aux_commitments.len() != vk.cs.num_aux_columns
            || self.aux_evals.len() != vk.cs.num_aux_columns
        {
            return Err(Error::IncompatibleParams);
        }

        // TODO: check h_evals

        if self.fixed_evals.len() != vk.cs.fixed_queries.len() {
            return Err(Error::IncompatibleParams);
        }

        if self.advice_evals.len() != vk.cs.advice_queries.len() {
            return Err(Error::IncompatibleParams);
        }

        self.permutations
            .as_ref()
            .map(|p| p.check_lengths(vk))
            .transpose()?;

        if self.lookup_proofs.len() != vk.cs.lookups.len() {
            return Err(Error::IncompatibleParams);
        }

        // TODO: check h_commitments

        if self.advice_commitments.len() != vk.cs.num_advice_columns {
            return Err(Error::IncompatibleParams);
        }

        Ok(())
    }

    /// Checks that this proof's h_evals are correct, and thus that all of the
    /// rules are satisfied.
    fn check_hx(
        &self,
        params: &'a Params<C>,
        vk: &VerifyingKey<C>,
        theta: ChallengeTheta<C::Scalar>,
        beta: ChallengeBeta<C::Scalar>,
        gamma: ChallengeGamma<C::Scalar>,
        y: ChallengeY<C::Scalar>,
        x: ChallengeX<C::Scalar>,
    ) -> Result<(), Error> {
        // x^n
        let xn = x.pow(&[params.n as u64, 0, 0, 0]);

        // TODO: bubble this error up
        // l_0(x)
        let l_0 = (*x - &C::Scalar::one()).invert().unwrap() // 1 / (x - 1)
            * &(xn - &C::Scalar::one()) // (x^n - 1) / (x - 1)
            * &vk.domain.get_barycentric_weight(); // l_0(x)

        // Compute the expected value of h(x)
        let expected_h_eval = std::iter::empty()
            // Evaluate the circuit using the custom gates provided
            .chain(vk.cs.gates.iter().map(|poly| {
                poly.evaluate(
                    &|index| self.fixed_evals[index],
                    &|index| self.advice_evals[index],
                    &|index| self.aux_evals[index],
                    &|a, b| a + &b,
                    &|a, b| a * &b,
                    &|a, scalar| a * &scalar,
                )
            }))
            .chain(
                self.permutations
                    .as_ref()
                    .map(|p| p.expressions(vk, &self.advice_evals, l_0, beta, gamma, x))
                    .into_iter()
                    .flatten(),
            )
            .chain(
                vk.cs
                    .lookups
                    .iter()
                    .zip(self.lookup_proofs.iter())
                    .flat_map(|(lookup, lookup_proof)| {
                        lookup_proof.evaluate_lookup_constraints(
                            &vk.cs,
                            *theta,
                            *beta,
                            *gamma,
                            l_0,
                            lookup,
                            &self.advice_evals,
                            &self.fixed_evals,
                            &self.aux_evals,
                        )
                    }),
            )
            .fold(C::Scalar::zero(), |h_eval, v| h_eval * &y + &v);

        // Compute h(x) from the prover
        let h_eval = self
            .h_evals
            .iter()
            .rev()
            .fold(C::Scalar::zero(), |acc, eval| acc * &xn + eval);

        // Did the prover commit to the correct polynomial?
        if expected_h_eval != (h_eval * &(xn - &C::Scalar::one())) {
            return Err(Error::ConstraintSystemFailure);
        }

        Ok(())
    }
}
