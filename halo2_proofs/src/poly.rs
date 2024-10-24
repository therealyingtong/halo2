//! Contains utilities for performing arithmetic over univariate polynomials in
//! various forms, including computing commitments to them and provably opening
//! the committed polynomials at arbitrary points.

use crate::arithmetic::parallelize;
use crate::helpers::SerdePrimeField;
use crate::plonk::Assigned;
use crate::SerdeFormat;
use ff::{PrimeField, WithSmallOrderMulGroup};
use group::ff::{BatchInvert, Field};
use halo2curves::fft::best_fft;
#[cfg(feature = "parallel-poly-read")]
use maybe_rayon::{iter::ParallelIterator, prelude::ParallelSliceMut};

use std::fmt::Debug;
use std::io;
use std::marker::PhantomData;
use std::ops::{Add, Deref, DerefMut, Div, Index, IndexMut, Mul, Range, RangeFrom, RangeFull, Sub};

/// Generic commitment scheme structures
pub mod commitment;
mod domain;
mod query;
mod strategy;

/// Inner product argument commitment scheme
pub mod ipa;

/// KZG commitment scheme
pub mod kzg;

#[cfg(test)]
mod multiopen_test;

pub use domain::*;
pub use query::{ProverQuery, VerifierQuery};
pub use strategy::{Guard, VerificationStrategy};

/// This is an error that could occur during proving or circuit synthesis.
// TODO: these errors need to be cleaned up
#[derive(Debug)]
pub enum Error {
    /// OpeningProof is not well-formed
    OpeningError,
    /// Caller needs to re-sample a point
    SamplingError,
}

/// The basis over which a polynomial is described.
pub trait Basis: Copy + Debug + Send + Sync {}

/// The polynomial is defined as coefficients
#[derive(Clone, Copy, Debug)]
pub struct Coeff;
impl Basis for Coeff {}

/// The polynomial is defined as coefficients of Lagrange basis polynomials
#[derive(Clone, Copy, Debug)]
pub struct LagrangeCoeff;
impl Basis for LagrangeCoeff {}

/// The polynomial is defined as coefficients of Lagrange basis polynomials in
/// an extended size domain which supports multiplication
#[derive(Clone, Copy, Debug)]
pub struct ExtendedLagrangeCoeff;
impl Basis for ExtendedLagrangeCoeff {}

/// Represents a univariate polynomial defined over a field and a particular
/// basis.
#[derive(Clone, Debug)]
pub struct Polynomial<F, B> {
    values: Vec<F>,
    _marker: PhantomData<B>,
}

impl<F> Polynomial<F, Coeff> {
    /// Polynomial from coeffs
    pub fn from_coeffs(values: Vec<F>) -> Self {
        Self {
            values,
            _marker: PhantomData,
        }
    }

    /// Coeffs of polynomial
    pub fn coeffs(&self) -> &[F] {
        &self.values
    }
}

impl<F, B> Index<usize> for Polynomial<F, B> {
    type Output = F;

    fn index(&self, index: usize) -> &F {
        self.values.index(index)
    }
}

impl<F, B> IndexMut<usize> for Polynomial<F, B> {
    fn index_mut(&mut self, index: usize) -> &mut F {
        self.values.index_mut(index)
    }
}

impl<F, B> Index<RangeFrom<usize>> for Polynomial<F, B> {
    type Output = [F];

    fn index(&self, index: RangeFrom<usize>) -> &[F] {
        self.values.index(index)
    }
}

impl<F, B> IndexMut<RangeFrom<usize>> for Polynomial<F, B> {
    fn index_mut(&mut self, index: RangeFrom<usize>) -> &mut [F] {
        self.values.index_mut(index)
    }
}

impl<F, B> Index<RangeFull> for Polynomial<F, B> {
    type Output = [F];

    fn index(&self, index: RangeFull) -> &[F] {
        self.values.index(index)
    }
}

impl<F, B> IndexMut<RangeFull> for Polynomial<F, B> {
    fn index_mut(&mut self, index: RangeFull) -> &mut [F] {
        self.values.index_mut(index)
    }
}

impl<F, B> Index<Range<usize>> for Polynomial<F, B> {
    type Output = [F];

    fn index(&self, index: Range<usize>) -> &[F] {
        self.values.index(index)
    }
}

impl<F, B> IndexMut<Range<usize>> for Polynomial<F, B> {
    fn index_mut(&mut self, index: Range<usize>) -> &mut [F] {
        self.values.index_mut(index)
    }
}

impl<F, B> Deref for Polynomial<F, B> {
    type Target = [F];

    fn deref(&self) -> &[F] {
        &self.values[..]
    }
}

impl<F, B> DerefMut for Polynomial<F, B> {
    fn deref_mut(&mut self) -> &mut [F] {
        &mut self.values[..]
    }
}

impl<F, B> Polynomial<F, B> {
    /// Iterate over the values, which are either in coefficient or evaluation
    /// form depending on the basis `B`.
    pub fn iter(&self) -> impl Iterator<Item = &F> {
        self.values.iter()
    }

    /// Iterate over the values mutably, which are either in coefficient or
    /// evaluation form depending on the basis `B`.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut F> {
        self.values.iter_mut()
    }

    /// Gets the size of this polynomial in terms of the number of
    /// coefficients used to describe it.
    pub fn num_coeffs(&self) -> usize {
        self.values.len()
    }
}

impl<F: SerdePrimeField, B> Polynomial<F, B> {
    /// Reads polynomial from buffer using `SerdePrimeField::read`.  
    #[cfg(feature = "parallel-poly-read")]
    pub(crate) fn read<R: io::Read>(reader: &mut R, format: SerdeFormat) -> io::Result<Self> {
        let mut poly_len = [0u8; 4];
        reader.read_exact(&mut poly_len)?;
        let poly_len = u32::from_be_bytes(poly_len) as usize;

        let repr_len = F::default().to_repr().as_ref().len();

        let mut new_vals = vec![0u8; poly_len * repr_len];
        reader.read_exact(&mut new_vals)?;

        // parallel read
        new_vals
            .par_chunks_mut(repr_len)
            .map(|chunk| {
                let mut chunk = io::Cursor::new(chunk);
                F::read(&mut chunk, format)
            })
            .collect::<io::Result<Vec<_>>>()
            .map(|values| Self {
                values,
                _marker: PhantomData,
            })
    }

    /// Reads polynomial from buffer using `SerdePrimeField::read`.  
    #[cfg(not(feature = "parallel-poly-read"))]
    pub(crate) fn read<R: io::Read>(reader: &mut R, format: SerdeFormat) -> io::Result<Self> {
        let mut poly_len = [0u8; 4];
        reader.read_exact(&mut poly_len)?;
        let poly_len = u32::from_be_bytes(poly_len);

        (0..poly_len)
            .map(|_| F::read(reader, format))
            .collect::<io::Result<Vec<_>>>()
            .map(|values| Self {
                values,
                _marker: PhantomData,
            })
    }

    /// Writes polynomial to buffer using `SerdePrimeField::write`.  
    pub(crate) fn write<W: io::Write>(
        &self,
        writer: &mut W,
        format: SerdeFormat,
    ) -> io::Result<()> {
        writer.write_all(&(self.values.len() as u32).to_be_bytes())?;
        for value in self.values.iter() {
            value.write(writer, format)?;
        }
        Ok(())
    }
}

pub(crate) fn batch_invert_assigned<F: Field>(
    assigned: Vec<Polynomial<Assigned<F>, LagrangeCoeff>>,
) -> Vec<Polynomial<F, LagrangeCoeff>> {
    let mut assigned_denominators: Vec<_> = assigned
        .iter()
        .map(|f| {
            f.iter()
                .map(|value| value.denominator())
                .collect::<Vec<_>>()
        })
        .collect();

    assigned_denominators
        .iter_mut()
        .flat_map(|f| {
            f.iter_mut()
                // If the denominator is trivial, we can skip it, reducing the
                // size of the batch inversion.
                .filter_map(|d| d.as_mut())
        })
        .batch_invert();

    assigned
        .iter()
        .zip(assigned_denominators)
        .map(|(poly, inv_denoms)| poly.invert(inv_denoms.into_iter().map(|d| d.unwrap_or(F::ONE))))
        .collect()
}

impl<F: Field> Polynomial<Assigned<F>, LagrangeCoeff> {
    pub(crate) fn invert(
        &self,
        inv_denoms: impl ExactSizeIterator<Item = F>,
    ) -> Polynomial<F, LagrangeCoeff> {
        assert_eq!(inv_denoms.len(), self.values.len());
        Polynomial {
            values: self
                .values
                .iter()
                .zip(inv_denoms)
                .map(|(a, inv_den)| a.numerator() * inv_den)
                .collect(),
            _marker: self._marker,
        }
    }
}

impl<'a, F: Field> Add<&'a Polynomial<F, LagrangeCoeff>> for Polynomial<F, LagrangeCoeff> {
    type Output = Polynomial<F, LagrangeCoeff>;

    fn add(mut self, rhs: &'a Polynomial<F, LagrangeCoeff>) -> Polynomial<F, LagrangeCoeff> {
        let coeffs_len = std::cmp::max(self.num_coeffs(), rhs.num_coeffs());
        self.values.resize(coeffs_len, F::ZERO);
        let mut rhs = rhs.clone();
        rhs.values.resize(coeffs_len, F::ZERO);

        parallelize(&mut self.values, |lhs, start| {
            for (lhs, rhs) in lhs.iter_mut().zip(rhs.values[start..].iter()) {
                *lhs += *rhs;
            }
        });

        self
    }
}

impl<'a, F: Field> Add<&'a Polynomial<F, ExtendedLagrangeCoeff>>
    for Polynomial<F, ExtendedLagrangeCoeff>
{
    type Output = Polynomial<F, ExtendedLagrangeCoeff>;

    fn add(
        mut self,
        rhs: &'a Polynomial<F, ExtendedLagrangeCoeff>,
    ) -> Polynomial<F, ExtendedLagrangeCoeff> {
        let coeffs_len = std::cmp::max(self.num_coeffs(), rhs.num_coeffs());
        self.values.resize(coeffs_len, F::ZERO);
        let mut rhs = rhs.clone();
        rhs.values.resize(coeffs_len, F::ZERO);

        parallelize(&mut self.values, |lhs, start| {
            for (lhs, rhs) in lhs.iter_mut().zip(rhs.values[start..].iter()) {
                *lhs += *rhs;
            }
        });

        self
    }
}

impl<'a, F: Field> Add<&'a Polynomial<F, Coeff>> for Polynomial<F, Coeff> {
    type Output = Polynomial<F, Coeff>;

    fn add(mut self, rhs: &'a Polynomial<F, Coeff>) -> Polynomial<F, Coeff> {
        self.strip_zeros();
        let mut rhs = rhs.clone();
        rhs.strip_zeros();

        let coeffs_len = std::cmp::max(self.num_coeffs(), rhs.num_coeffs());
        self.values.resize(coeffs_len, F::ZERO);
        let mut rhs = rhs.clone();
        rhs.values.resize(coeffs_len, F::ZERO);

        parallelize(&mut self.values, |lhs, start| {
            for (lhs, rhs) in lhs.iter_mut().zip(rhs.values[start..].iter()) {
                *lhs += *rhs;
            }
        });

        self
    }
}

impl<'a, F: Field> Sub<&'a Polynomial<F, LagrangeCoeff>> for Polynomial<F, LagrangeCoeff> {
    type Output = Polynomial<F, LagrangeCoeff>;

    fn sub(mut self, rhs: &'a Polynomial<F, LagrangeCoeff>) -> Polynomial<F, LagrangeCoeff> {
        let coeffs_len = std::cmp::max(self.num_coeffs(), rhs.num_coeffs());
        self.values.resize(coeffs_len, F::ZERO);
        let mut rhs = rhs.clone();
        rhs.values.resize(coeffs_len, F::ZERO);

        parallelize(&mut self.values, |lhs, start| {
            for (lhs, rhs) in lhs.iter_mut().zip(rhs.values[start..].iter()) {
                *lhs -= *rhs;
            }
        });

        self
    }
}

impl<'a, F: Field> Sub<&'a Polynomial<F, ExtendedLagrangeCoeff>>
    for Polynomial<F, ExtendedLagrangeCoeff>
{
    type Output = Polynomial<F, ExtendedLagrangeCoeff>;

    fn sub(
        mut self,
        rhs: &'a Polynomial<F, ExtendedLagrangeCoeff>,
    ) -> Polynomial<F, ExtendedLagrangeCoeff> {
        let coeffs_len = std::cmp::max(self.num_coeffs(), rhs.num_coeffs());
        self.values.resize(coeffs_len, F::ZERO);
        let mut rhs = rhs.clone();
        rhs.values.resize(coeffs_len, F::ZERO);

        parallelize(&mut self.values, |lhs, start| {
            for (lhs, rhs) in lhs.iter_mut().zip(rhs.values[start..].iter()) {
                *lhs -= *rhs;
            }
        });

        self
    }
}

impl<'a, F: Field> Sub<&'a Polynomial<F, Coeff>> for Polynomial<F, Coeff> {
    type Output = Polynomial<F, Coeff>;

    fn sub(mut self, rhs: &'a Polynomial<F, Coeff>) -> Polynomial<F, Coeff> {
        self.strip_zeros();
        let mut rhs = rhs.clone();
        rhs.strip_zeros();

        let coeffs_len = std::cmp::max(self.num_coeffs(), rhs.num_coeffs());
        self.values.resize(coeffs_len, F::ZERO);
        let mut rhs = rhs.clone();
        rhs.values.resize(coeffs_len, F::ZERO);

        parallelize(&mut self.values, |lhs, start| {
            for (lhs, rhs) in lhs.iter_mut().zip(rhs.values[start..].iter()) {
                *lhs -= *rhs;
            }
        });

        self
    }
}

impl<F: Field> Polynomial<F, LagrangeCoeff> {
    /// Rotates the values in a Lagrange basis polynomial by `Rotation`
    pub fn rotate(&self, rotation: Rotation) -> Polynomial<F, LagrangeCoeff> {
        let mut values = self.values.clone();
        if rotation.0 < 0 {
            values.rotate_right((-rotation.0) as usize);
        } else {
            values.rotate_left(rotation.0 as usize);
        }
        Polynomial {
            values,
            _marker: PhantomData,
        }
    }
}

impl<F: Field> Mul<F> for Polynomial<F, LagrangeCoeff> {
    type Output = Polynomial<F, LagrangeCoeff>;

    fn mul(mut self, rhs: F) -> Polynomial<F, LagrangeCoeff> {
        if rhs == F::ZERO {
            return Polynomial {
                values: vec![F::ZERO; self.len()],
                _marker: PhantomData,
            };
        }
        if rhs == F::ONE {
            return self;
        }

        parallelize(&mut self.values, |lhs, _| {
            for lhs in lhs.iter_mut() {
                *lhs *= rhs;
            }
        });

        self
    }
}

impl<F: Field> Mul<F> for Polynomial<F, Coeff> {
    type Output = Polynomial<F, Coeff>;

    fn mul(mut self, rhs: F) -> Polynomial<F, Coeff> {
        self.strip_zeros();
        if rhs == F::ZERO {
            return Polynomial {
                values: vec![F::ZERO; self.len()],
                _marker: PhantomData,
            };
        }
        if rhs == F::ONE {
            return self;
        }

        parallelize(&mut self.values, |lhs, _| {
            for lhs in lhs.iter_mut() {
                *lhs *= rhs;
            }
        });

        self
    }
}

impl<F: Field> Polynomial<F, Coeff> {
    /// Returns true if the polynomial is the zero polynomial
    pub fn is_zero(&self) -> bool {
        self.coeffs().iter().fold(F::ZERO, |acc, val| acc + val) == F::ZERO
    }

    /// Remove trailing zeros (higher-degree coeffs)
    pub fn strip_zeros(&mut self) {
        while self.values.last().unwrap() == &F::ZERO && self.values.len() > 1 {
            self.values.pop();
        }
    }
}

impl<F: PrimeField + WithSmallOrderMulGroup<3>> Polynomial<F, Coeff> {
    /// Divide with quotient and remainder
    pub fn divide_with_q_and_r(&self, divisor: Self) -> (Self, Self) {
        let mut poly = self.clone();
        let mut divisor = divisor.clone();
        poly.strip_zeros();
        divisor.strip_zeros();

        if self.is_zero() || self.num_coeffs() < divisor.num_coeffs() {
            return (
                Polynomial {
                    values: vec![F::ZERO],
                    _marker: PhantomData,
                },
                self.clone(),
            );
        }

        let quotient_num_coeffs = poly.num_coeffs() - divisor.num_coeffs() + 1;

        let mut quotient = vec![F::ZERO; quotient_num_coeffs];
        let mut remainder = poly.clone();

        // Can unwrap here because we know self is not zero.
        let divisor_leading_inv = divisor.last().unwrap().invert().unwrap();
        while !remainder.is_zero() && remainder.num_coeffs() >= divisor.num_coeffs() {
            let cur_q_coeff = *remainder.values.last().unwrap() * divisor_leading_inv;
            let cur_q_degree = (remainder.num_coeffs() - 1) - (divisor.num_coeffs() - 1);

            quotient[cur_q_degree] = cur_q_coeff;

            for (i, div_coeff) in divisor.iter().enumerate() {
                remainder[cur_q_degree + i] -= &(cur_q_coeff * div_coeff);
            }
            remainder.strip_zeros();
        }

        let mut quotient = Polynomial::from_coeffs(quotient);
        quotient.strip_zeros();

        (quotient, remainder)
    }
}

impl<F: PrimeField + WithSmallOrderMulGroup<3>> Div<Polynomial<F, Coeff>> for Polynomial<F, Coeff> {
    type Output = Self;

    fn div(self, rhs: Polynomial<F, Coeff>) -> Self::Output {
        self.divide_with_q_and_r(rhs).0
    }
}

impl<F: Field + WithSmallOrderMulGroup<3>> Mul<Polynomial<F, Coeff>> for Polynomial<F, Coeff> {
    type Output = Polynomial<F, Coeff>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Polynomial<F, Coeff>) -> Self::Output {
        let mut poly = self.clone();
        poly.strip_zeros();

        let mut rhs = rhs.clone();
        rhs.strip_zeros();

        if poly.num_coeffs() == 1 {
            return rhs * poly.values[0];
        }
        if rhs.num_coeffs() == 1 {
            return poly * rhs.values[0];
        }

        // Extend both polynomials coefficient vectors to their nearest power-of-2 resultant degree
        let resulting_degree = (poly.num_coeffs() - 1) + (rhs.num_coeffs() - 1);
        let num_coeffs = resulting_degree + 1;
        let n = num_coeffs.next_power_of_two();
        poly.values.resize(n, F::ZERO);
        rhs.values.resize(n, F::ZERO);

        // Take the forward FFT of the two polynomials
        let log_n = n.ilog2() as u32;
        let domain: EvaluationDomain<F> = EvaluationDomain::new(1, log_n);
        best_fft(&mut poly.values, domain.get_omega(), domain.k());
        best_fft(&mut rhs.values, domain.get_omega(), domain.k());

        // Multiply the two polynomials in the FFT image
        let mut mul_res = poly
            .values
            .iter()
            .zip(rhs.values.iter())
            .map(|(&val, &rhs)| val * rhs)
            .collect::<Vec<_>>();

        // Take the inverse FFT of the result to get the coefficients of the product
        best_fft(&mut mul_res, domain.get_omega_inv(), domain.k());
        let mut mul_res = Polynomial::from_coeffs(mul_res);
        mul_res.strip_zeros();

        let n_inv = F::from(domain.get_n()).invert().unwrap();
        mul_res.iter_mut().for_each(|v| *v *= n_inv);

        mul_res
    }
}

impl<'a, F: Field, B: Basis> Sub<F> for &'a Polynomial<F, B> {
    type Output = Polynomial<F, B>;

    fn sub(self, rhs: F) -> Polynomial<F, B> {
        let mut res = self.clone();
        res.values[0] -= rhs;
        res
    }
}

impl<'a, F: Field> Add<F> for &'a Polynomial<F, LagrangeCoeff> {
    type Output = Polynomial<F, LagrangeCoeff>;

    fn add(self, rhs: F) -> Polynomial<F, LagrangeCoeff> {
        let mut res = self.clone();
        res.values[0] += rhs;
        res
    }
}

impl<'a, F: Field> Add<F> for &'a Polynomial<F, Coeff> {
    type Output = Polynomial<F, Coeff>;

    fn add(self, rhs: F) -> Polynomial<F, Coeff> {
        let mut res = self.clone();
        res.strip_zeros();
        res.values[0] += rhs;
        res
    }
}

/// Describes the relative rotation of a vector. Negative numbers represent
/// reverse (leftmost) rotations and positive numbers represent forward (rightmost)
/// rotations. Zero represents no rotation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Rotation(pub i32);

impl Rotation {
    /// The current location in the evaluation domain
    pub fn cur() -> Rotation {
        Rotation(0)
    }

    /// The previous location in the evaluation domain
    pub fn prev() -> Rotation {
        Rotation(-1)
    }

    /// The next location in the evaluation domain
    pub fn next() -> Rotation {
        Rotation(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff::Field;
    use halo2curves::bls12381::Fr;
    use proptest::prelude::Rng;
    use rand_core::OsRng;

    fn random_poly<F: Field>(num_coeffs: usize) -> Polynomial<F, Coeff> {
        let mut rng = OsRng;
        // Sample a random degree below the bound
        let mut coeffs = Vec::with_capacity(num_coeffs);
        for _ in 0..num_coeffs {
            // Sample a random coefficient
            coeffs.push(F::random(&mut rng));
        }

        Polynomial::from_coeffs(coeffs)
    }

    #[test]
    fn test_poly_div() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let num_coeffs = 256;
        let a = random_poly::<Fr>(num_coeffs);
        let b = random_poly::<Fr>(rng.gen_range(1..num_coeffs));

        let ab = a.clone() * b.clone();
        let res_a = ab.clone() / b.clone();
        let res_b = ab.clone() / a.clone();

        assert_eq!(a.coeffs(), res_a.coeffs());
        assert_eq!(b.coeffs(), res_b.coeffs());

        let r = &b - Fr::one();
        let abr = ab.clone() + &r;
        let (res_a, res_r) = abr.divide_with_q_and_r(b.clone());
        assert_eq!((&a + Fr::one()).coeffs(), res_a.coeffs());
        assert_eq!(-Fr::one(), res_r.coeffs()[0]);

        let r = &a - Fr::one();
        let abr = ab + &r;
        let (res_b, res_r) = abr.divide_with_q_and_r(a);
        assert_eq!((&b + Fr::one()).coeffs(), res_b.coeffs());
        assert_eq!(-Fr::one(), res_r.coeffs()[0]);
    }
}
