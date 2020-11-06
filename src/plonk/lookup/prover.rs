use super::super::{
    circuit::{Advice, Any, Column, Fixed},
    ProvingKey,
};
use super::{Lookup, Permuted, Product, Proof};
use crate::arithmetic::{eval_polynomial, parallelize, BatchInvert, Curve, CurveAffine, Field};
use crate::poly::{
    commitment::{Blind, Params},
    EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial, Rotation,
};
use std::collections::BTreeMap;

pub struct LookupData<C: CurveAffine> {
    pub lookup: Lookup,
    pub permuted: Option<Permuted<C>>,
    pub product: Option<Product<C>>,
}

impl<C: CurveAffine> LookupData<C> {
    pub fn new(lookup: &Lookup) -> Self {
        assert_eq!(lookup.input_columns.len(), lookup.table_columns.len());
        LookupData {
            lookup: lookup.clone(),
            permuted: None,
            product: None,
        }
    }
}
