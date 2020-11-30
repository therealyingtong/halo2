use super::Error;
use crate::{
    arithmetic::{eval_polynomial, CurveAffine},
    plonk::{ChallengeX, ChallengeY, ProvingKey},
    poly::{
        commitment::{Blind, Params},
        Coeff, ExtendedLagrangeCoeff, Polynomial,
    },
    transcript::{Hasher, Transcript},
};

/// A vanishing argument.
pub(crate) struct Argument<C: CurveAffine> {
    h_commitments: Vec<C>,
}

pub(crate) struct Constructed<C: CurveAffine> {
    h_pieces: Vec<Polynomial<C::Scalar, Coeff>>,
    h_commitments: Vec<C>,
}

pub(crate) struct Evaluated<C: CurveAffine> {
    h_evals: Vec<C::Scalar>,
}

impl<C: CurveAffine> Argument<C> {
    pub(crate) fn construct<HBase: Hasher<C::Base>, HScalar: Hasher<C::Scalar>>(
        params: &Params<C>,
        pk: &ProvingKey<C>,
        expressions: impl Iterator<Item = Polynomial<C::Scalar, ExtendedLagrangeCoeff>>,
        transcript: &mut Transcript<C, HBase, HScalar>,
    ) -> Constructed<C> {
        let domain = &pk.vk.domain;

        // Obtain challenge for keeping all separate gates linearly independent
        let y = ChallengeY::<C::Scalar>::get(&mut transcript);

        // Evaluate the h(X) polynomial's constraint system expressions for the constraints provided
        let h_poly = expressions.fold(domain.empty_extended(), |h_poly, v| h_poly * *y + &v);

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

        // Hash each h(X) piece
        for c in h_commitments.iter() {
            transcript
                .absorb_point(c)
                .map_err(|_| Error::TranscriptError)?;
        }

        Constructed {
            h_pieces,
            h_commitments,
        }
    }
}

impl<C: CurveAffine> Constructed<C> {
    pub(crate) fn evaluate<HBase: Hasher<C::Base>, HScalar: Hasher<C::Scalar>>(
        self,
        pk: &ProvingKey<C>,
        x: ChallengeX<C::Scalar>,
        transcript: &mut Transcript<C, HBase, HScalar>,
    ) -> Evaluated<C> {
        let domain = &pk.vk.domain;

        let h_evals: Vec<_> = self
            .h_pieces
            .iter()
            .map(|poly| eval_polynomial(poly, *x))
            .collect();

        // Hash each advice evaluation
        for eval in h_evals {
            transcript.absorb_scalar(eval);
        }

        Evaluated { h_evals }
    }
}
