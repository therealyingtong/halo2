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

    pub fn construct_permuted(
        &mut self,
        pk: &ProvingKey<C>,
        params: &Params<C>,
        domain: &EvaluationDomain<C::Scalar>,
        theta: C::Scalar,
        advice_values: &[Polynomial<C::Scalar, LagrangeCoeff>],
        fixed_values: &[Polynomial<C::Scalar, LagrangeCoeff>],
    ) -> Permuted<C> {
        // Values of input columns involved in the lookup
        let unpermuted_input_values: Vec<Polynomial<C::Scalar, LagrangeCoeff>> = self
            .lookup
            .input_columns
            .iter()
            .map(|&input| match input.column_type {
                Any::Advice => advice_values[input.index].clone(),
                Any::Fixed => fixed_values[input.index].clone(),
                _ => unreachable!(),
            })
            .collect();

        // Compressed version of input columns
        let compressed_input_value = unpermuted_input_values
            .iter()
            .fold(domain.empty_lagrange(), |acc, input| acc * theta + input);

        // Values of table columns involved in the lookup
        let unpermuted_table_values: Vec<Polynomial<C::Scalar, LagrangeCoeff>> = self
            .lookup
            .table_columns
            .iter()
            .map(|&table| match table.column_type {
                Any::Advice => advice_values[table.index].clone(),
                Any::Fixed => fixed_values[table.index].clone(),
                _ => unreachable!(),
            })
            .collect();

        // Compressed version of table columns
        let compressed_table_value = unpermuted_table_values
            .iter()
            .fold(domain.empty_lagrange(), |acc, table| acc * theta + table);

        // Permute compressed (InputColumn, TableColumn) pair
        let (permuted_input_value, permuted_table_value) =
            LookupData::<C>::permute_column_pair(&compressed_input_value, &compressed_table_value);

        // Construct Permuted struct
        let permuted_input_poly = pk.vk.domain.lagrange_to_coeff(permuted_input_value.clone());
        let permuted_input_coset = pk
            .vk
            .domain
            .coeff_to_extended(permuted_input_poly.clone(), Rotation::default());
        let permuted_input_inv_coset = pk
            .vk
            .domain
            .coeff_to_extended(permuted_input_poly.clone(), Rotation(-1));

        let permuted_input_blind = Blind(C::Scalar::random());
        let permuted_input_commitment = params
            .commit_lagrange(&permuted_input_value, permuted_input_blind)
            .to_affine();

        let permuted_table_poly = pk.vk.domain.lagrange_to_coeff(permuted_table_value.clone());
        let permuted_table_coset = pk
            .vk
            .domain
            .coeff_to_extended(permuted_table_poly.clone(), Rotation::default());
        let permuted_table_blind = Blind(C::Scalar::random());
        let permuted_table_commitment = params
            .commit_lagrange(&permuted_table_value, permuted_table_blind)
            .to_affine();

        let permuted = Permuted {
            permuted_input_value,
            permuted_input_poly,
            permuted_input_coset,
            permuted_input_inv_coset,
            permuted_input_blind,
            permuted_input_commitment,
            permuted_table_value,
            permuted_table_poly,
            permuted_table_coset,
            permuted_table_blind,
            permuted_table_commitment,
        };

        self.permuted = Some(permuted.clone());
        permuted
    }

    fn permute_column_pair(
        input_value: &Polynomial<C::Scalar, LagrangeCoeff>,
        table_value: &Polynomial<C::Scalar, LagrangeCoeff>,
    ) -> (
        Polynomial<C::Scalar, LagrangeCoeff>,
        Polynomial<C::Scalar, LagrangeCoeff>,
    ) {
        let mut input_coeffs = input_value.get_values().to_vec();
        let table_coeffs = table_value.get_values().to_vec();

        // Sort input lookup column values
        input_coeffs.sort();
        let permuted_input_value = Polynomial::new(input_coeffs.to_vec());

        // A BTreeMap of each unique element in the table column and its count
        let mut leftover_table_map: BTreeMap<C::Scalar, u32> =
            table_coeffs.iter().fold(BTreeMap::new(), |mut acc, coeff| {
                *acc.entry(*coeff).or_insert(0) += 1;
                acc
            });
        let mut repeated_input_rows = vec![];
        let mut permuted_table_coeffs = vec![C::Scalar::zero(); table_coeffs.len()];

        for row in 0..input_coeffs.len() {
            let input_value = input_coeffs[row];

            // If this is the first occurence of `input_value` in the input column
            if row == 0 || input_value != input_coeffs[row - 1] {
                permuted_table_coeffs[row] = input_value;
                // Remove one instance of input_value from leftover_table_map
                if let Some(count) = leftover_table_map.get_mut(&input_value) {
                    *count -= 1;
                } else {
                    // Panic if input_value not found
                    panic!("Input value not found in table.");
                }
            // If input value is repeated
            } else {
                repeated_input_rows.push(row);
            }
        }

        // Convert BTreeMap back into vector, using appropriate counts for each key
        let leftover_table_coeffs: Vec<C::Scalar> = leftover_table_map.iter().fold(
            Vec::with_capacity(repeated_input_rows.len()),
            |mut acc, (coeff, count)| {
                acc.extend(vec![*coeff; *count as usize]);
                acc
            },
        );

        // Populate permuted table at unfilled rows with leftover table elements
        for (idx, &row) in repeated_input_rows.iter().enumerate() {
            permuted_table_coeffs[row] = leftover_table_coeffs[idx];
        }

        let permuted_table_value = Polynomial::new(permuted_table_coeffs.to_vec());

        (permuted_input_value, permuted_table_value)
    }
}
