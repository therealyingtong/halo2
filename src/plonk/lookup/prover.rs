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
                    assert!(*count > 0);
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

        // Populate permuted table at unfilled rows with leftover table elements
        for (coeff, count) in leftover_table_map.iter() {
            for _ in 0..*count {
                permuted_table_coeffs[repeated_input_rows.remove(0) as usize] = *coeff;
            }
        }

        let permuted_table_value = Polynomial::new(permuted_table_coeffs.to_vec());

        (permuted_input_value, permuted_table_value)
    }

    pub fn construct_product(
        &mut self,
        pk: &ProvingKey<C>,
        params: &Params<C>,
        beta: C::Scalar,
        gamma: C::Scalar,
        theta: C::Scalar,
        advice_values: &[Polynomial<C::Scalar, LagrangeCoeff>],
        fixed_values: &[Polynomial<C::Scalar, LagrangeCoeff>],
    ) -> Product<C> {
        let permuted = self.permuted.clone().unwrap();
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

        // Goal is to compute the products of fractions
        //
        // (a_1(\omega^i) + \theta a_2(\omega^i) + ... + beta)(s_1(\omega^i) + \theta(\omega^i) + ... + \gamma) /
        // (a'(\omega^i) + \beta)(s'(\omega^i) + \gamma)
        //
        // where a_j(X) is the jth input column in this lookup,
        // where a'(X) is the compression of the permuted input columns,
        // q_j(X) is the jth table column in this lookup,
        // q'(X) is the compression of the permuted table columns,
        // and i is the ith row of the column.
        let mut lookup_product = vec![C::Scalar::one(); params.n as usize];

        // Denominator uses the permuted input column and permuted table column
        parallelize(&mut lookup_product, |lookup_product, start| {
            for ((lookup_product, permuted_input_value), permuted_table_value) in lookup_product
                .iter_mut()
                .zip(permuted.permuted_input_value[start..].iter())
                .zip(permuted.permuted_table_value[start..].iter())
            {
                *lookup_product *= &(beta + permuted_input_value);
                *lookup_product *= &(gamma + permuted_table_value);
            }
        });

        // Batch invert to obtain the denominators for the lookup product
        // polynomials
        lookup_product.iter_mut().batch_invert();

        // Finish the computation of the entire fraction by computing the numerators
        // (a_1(X) + \theta a_2(X) + ... + \beta) (s_1(X) + \theta s_2(X) + ... + \gamma)
        // Compress unpermuted InputColumns
        let mut input_term = vec![C::Scalar::zero(); params.n as usize];
        for unpermuted_input_value in unpermuted_input_values.iter() {
            parallelize(&mut input_term, |input_term, start| {
                for (input_term, input_value) in input_term
                    .iter_mut()
                    .zip(unpermuted_input_value.get_values()[start..].iter())
                {
                    *input_term *= &theta;
                    *input_term += input_value;
                }
            });
        }

        // Compress unpermuted TableColumns
        let mut table_term = vec![C::Scalar::zero(); params.n as usize];
        for unpermuted_table_value in unpermuted_table_values.iter() {
            parallelize(&mut table_term, |table_term, start| {
                for (table_term, fixed_value) in table_term
                    .iter_mut()
                    .zip(unpermuted_table_value.get_values()[start..].iter())
                {
                    *table_term *= &theta;
                    *table_term += fixed_value;
                }
            });
        }

        // Add \beta and \gamma offsets
        parallelize(&mut lookup_product, |product, start| {
            for ((product, input_term), table_term) in product
                .iter_mut()
                .zip(input_term[start..].iter())
                .zip(table_term[start..].iter())
            {
                *product *= &(*input_term + &beta);
                *product *= &(*table_term + &gamma);
            }
        });

        // The product vector is a vector of products of fractions of the form
        //
        // (a_1(\omega^i) + \theta a_2(\omega^i) + ... + \beta)(s_1(\omega^i) + \theta s_2(\omega^i) + ... + \gamma)/
        // (a'(\omega^i) + \beta) (s'(\omega^i) + \gamma)
        //
        // where a_j(\omega^i) is the jth input column in this lookup,
        // a'j(\omega^i) is the permuted input column,
        // s_j(\omega^i) is the jth table column in this lookup,
        // s'(\omega^i) is the permuted table column,
        // and i is the ith row of the column.

        // Compute the evaluations of the lookup product polynomial
        // over our domain, starting with z[0] = 1
        let mut z = vec![C::Scalar::one()];
        for row in 1..(params.n as usize) {
            let mut tmp = z[row - 1];
            tmp *= &lookup_product[row];
            z.push(tmp);
        }
        let z = pk.vk.domain.lagrange_from_vec(z);

        // Check lagrange form of product is correctly constructed over the domain
        // z'(X) (a'(X) + \beta) (s'(X) + \gamma)
        // - z'(\omega^{-1} X) (a_1(X) + \theta a_2(X) + ... + \beta) (s_1(X) + \theta s_2(X) + ... + \gamma)
        let n = params.n as usize;

        for i in 0..n {
            let prev_idx = (n + i - 1) % n;

            let mut left = z.get_values().clone()[i];
            let permuted_input_value = &permuted.permuted_input_value.get_values()[i];

            let permuted_table_value = &permuted.permuted_table_value.get_values()[i];

            left *= &(beta + permuted_input_value);
            left *= &(gamma + permuted_table_value);

            let mut right = z.get_values().clone()[prev_idx];
            let mut input_term = unpermuted_input_values
                .iter()
                .fold(C::Scalar::zero(), |acc, input| {
                    acc * &theta + &input.get_values()[i]
                });

            let mut table_term = unpermuted_table_values
                .iter()
                .fold(C::Scalar::zero(), |acc, table| {
                    acc * &theta + &table.get_values()[i]
                });

            input_term += &beta;
            table_term += &gamma;
            right *= &(input_term * &table_term);

            assert_eq!(left, right);
        }

        let product_blind = Blind(C::Scalar::random());
        let product_commitment = params.commit_lagrange(&z, product_blind).to_affine();
        let z = pk.vk.domain.lagrange_to_coeff(z);
        let product_coset = pk
            .vk
            .domain
            .coeff_to_extended(z.clone(), Rotation::default());
        let product_inv_coset = pk.vk.domain.coeff_to_extended(z.clone(), Rotation(-1));

        let product = Product::<C> {
            product_poly: z.clone(),
            product_coset,
            product_inv_coset,
            product_commitment,
            product_blind,
        };

        self.product = Some(product.clone());
        product
    }

    pub fn construct_constraints(
        &self,
        pk: &ProvingKey<C>,
        beta: C::Scalar,
        gamma: C::Scalar,
        theta: C::Scalar,
        advice_cosets: &[Polynomial<C::Scalar, ExtendedLagrangeCoeff>],
        fixed_cosets: &[Polynomial<C::Scalar, ExtendedLagrangeCoeff>],
    ) -> Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>> {
        let permuted = self.permuted.clone().unwrap();
        let product = self.product.clone().unwrap();
        let unpermuted_input_cosets: Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>> = self
            .lookup
            .input_columns
            .iter()
            .map(|&input| match input.column_type {
                Any::Advice => advice_cosets[pk
                    .vk
                    .cs
                    .get_advice_query_index(Column::<Advice>::from(input), 0)]
                .clone(),
                Any::Fixed => fixed_cosets[pk
                    .vk
                    .cs
                    .get_fixed_query_index(Column::<Fixed>::from(input), 0)]
                .clone(),
                _ => unreachable!(),
            })
            .collect();

        let unpermuted_table_cosets: Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>> = self
            .lookup
            .table_columns
            .iter()
            .map(|&table| match table.column_type {
                Any::Advice => advice_cosets[pk
                    .vk
                    .cs
                    .get_advice_query_index(Column::<Advice>::from(table), 0)]
                .clone(),
                Any::Fixed => fixed_cosets[pk
                    .vk
                    .cs
                    .get_fixed_query_index(Column::<Fixed>::from(table), 0)]
                .clone(),
                _ => unreachable!(),
            })
            .collect();

        let mut constraints: Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>> =
            Vec::with_capacity(4);

        // l_0(X) * (1 - z'(X)) = 0
        {
            let mut first_product_poly = pk.vk.domain.empty_extended();
            parallelize(&mut first_product_poly, |first_product_poly, start| {
                for ((first_product_poly, product), l0) in first_product_poly
                    .iter_mut()
                    .zip(product.product_coset[start..].iter())
                    .zip(pk.l0[start..].iter())
                {
                    *first_product_poly += &(*l0 * &(C::Scalar::one() - product));
                }
            });
            constraints.push(first_product_poly);
        }

        // z'(X) (a'(X) + \beta) (s'(X) + \gamma)
        // - z'(\omega^{-1} X) (a_1(X) + \theta a_2(X) + ... + \beta) (s_1(X) + \theta s_2(X) + ... + \gamma)
        {
            // z'(X) (a'(X) + \beta) (s'(X) + \gamma)
            let mut left = product.product_coset.clone();
            parallelize(&mut left, |left, start| {
                for ((left, permuted_input), permuted_table) in left
                    .iter_mut()
                    .zip(permuted.permuted_input_coset[start..].iter())
                    .zip(permuted.permuted_table_coset[start..].iter())
                {
                    *left *= &(*permuted_input + &beta);
                    *left *= &(*permuted_table + &gamma);
                }
            });

            //  z'(\omega^{-1} X) (a_1(X) + \theta a_2(X) + ... + \beta) (s_1(X) + \theta s_2(X) + ... + \gamma)
            let mut right = product.product_inv_coset.clone();
            let mut input_terms = pk.vk.domain.empty_extended();

            // Compress the unpermuted Input columns
            for input in unpermuted_input_cosets.iter() {
                // (a_1(X) + \theta a_2(X) + ...)
                parallelize(&mut input_terms, |input_term, start| {
                    for (input_term, input) in input_term.iter_mut().zip(input[start..].iter()) {
                        *input_term *= &theta;
                        *input_term += input;
                    }
                });
            }

            let mut table_terms = pk.vk.domain.empty_extended();
            // Compress the unpermuted Table columns
            for table in unpermuted_table_cosets.iter() {
                //  (s_1(X) + \theta s_2(X) + ...)
                parallelize(&mut table_terms, |table_term, start| {
                    for (table_term, table) in table_term.iter_mut().zip(table[start..].iter()) {
                        *table_term *= &theta;
                        *table_term += table;
                    }
                });
            }

            // Add \beta and \gamma offsets
            parallelize(&mut right, |right, start| {
                for ((right, input_term), table_term) in right
                    .iter_mut()
                    .zip(input_terms[start..].iter())
                    .zip(table_terms[start..].iter())
                {
                    *right *= &(*input_term + &beta);
                    *right *= &(*table_term + &gamma);
                }
            });

            constraints.push(left - &right);
        }

        // Check that the first values in the permuted input column and permuted
        // fixed column are the same.
        // l_0(X) * (a'(X) - s'(X)) = 0
        {
            let mut first_lookup_poly =
                permuted.permuted_input_coset.clone() - &permuted.permuted_table_coset.clone();
            first_lookup_poly = first_lookup_poly * &pk.l0;
            constraints.push(first_lookup_poly);
        }

        // Check that each value in the permuted lookup input column is either
        // equal to the value above it, or the value at the same index in the
        // permuted table column.
        // (a′(X)−s′(X))⋅(a′(X)−a′(\omega{-1} X)) = 0
        {
            let mut lookup_poly =
                permuted.permuted_input_coset.clone() - &permuted.permuted_table_coset;
            lookup_poly = lookup_poly
                * &(permuted.permuted_input_coset.clone() - &permuted.permuted_input_inv_coset);
            constraints.push(lookup_poly);
        }

        constraints
    }

    pub fn construct_proof(
        &mut self,
        domain: &EvaluationDomain<C::Scalar>,
        point: C::Scalar,
    ) -> Proof<C> {
        let product = self.product.clone().unwrap();
        let permuted = self.permuted.clone().unwrap();

        let product_eval: C::Scalar = eval_polynomial(&product.product_poly.get_values(), point);

        let product_inv_eval: C::Scalar = eval_polynomial(
            &product.product_poly.get_values(),
            domain.rotate_omega(point, Rotation(-1)),
        );

        let permuted_input_eval: C::Scalar = eval_polynomial(&permuted.permuted_input_poly, point);
        let permuted_input_inv_eval: C::Scalar = eval_polynomial(
            &permuted.permuted_input_poly,
            domain.rotate_omega(point, Rotation(-1)),
        );
        let permuted_table_eval: C::Scalar = eval_polynomial(&permuted.permuted_table_poly, point);

        Proof {
            product_commitment: product.product_commitment,
            product_eval,
            product_inv_eval,
            permuted_input_commitment: permuted.permuted_input_commitment,
            permuted_table_commitment: permuted.permuted_table_commitment,
            permuted_input_eval,
            permuted_input_inv_eval,
            permuted_table_eval,
        }
    }
}
