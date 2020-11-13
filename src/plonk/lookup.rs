use super::circuit::{Any, Column};
use crate::arithmetic::CurveAffine;
use crate::poly::{commitment::Blind, Coeff, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial};
pub mod prover;
pub mod verifier;

#[derive(Clone, Debug)]
pub struct Lookup {
    pub input_columns: Vec<Column<Any>>,
    pub table_columns: Vec<Column<Any>>,
}

#[derive(Clone, Debug)]
pub struct Proof<C: CurveAffine> {
    pub product_commitment: C,
    pub product_eval: C::Scalar,
    pub product_inv_eval: C::Scalar,
    pub permuted_input_commitment: C,
    pub permuted_table_commitment: C,
    pub permuted_input_eval: C::Scalar,
    pub permuted_input_inv_eval: C::Scalar,
    pub permuted_table_eval: C::Scalar,
}

#[derive(Clone, Debug)]
pub struct Permuted<C: CurveAffine> {
    pub permuted_input_value: Polynomial<C::Scalar, LagrangeCoeff>,
    pub permuted_input_poly: Polynomial<C::Scalar, Coeff>,
    pub permuted_input_coset: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    pub permuted_input_inv_coset: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    pub permuted_input_blind: Blind<C::Scalar>,
    pub permuted_input_commitment: C,
    pub permuted_table_value: Polynomial<C::Scalar, LagrangeCoeff>,
    pub permuted_table_poly: Polynomial<C::Scalar, Coeff>,
    pub permuted_table_coset: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    pub permuted_table_blind: Blind<C::Scalar>,
    pub permuted_table_commitment: C,
}

#[derive(Clone, Debug)]
pub struct Product<C: CurveAffine> {
    pub product_poly: Polynomial<C::Scalar, Coeff>,
    pub product_coset: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    pub product_inv_coset: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    pub product_commitment: C,
    pub product_blind: Blind<C::Scalar>,
}

impl Lookup {
    pub fn new(input_columns: &[Column<Any>], table_columns: &[Column<Any>]) -> Self {
        assert_eq!(input_columns.len(), table_columns.len());
        Lookup {
            input_columns: input_columns.to_vec(),
            table_columns: table_columns.to_vec(),
        }
    }
}