// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/util/polyarithsmallmod.h"
#include "seal/util/uintarith.h"
#include "seal/util/uintcore.h"
#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

#ifdef SEAL_USE_INTEL_HEXL
#include "hexl/hexl.hpp"
#endif

using namespace std;

namespace seal
{
    namespace util
    {
        #if defined(__riscv_v_intrinsic)
        
            void add_uint64_vector(vuint64m4_t a, vuint64m4_t b, vuint64m4_t *sum, vuint64m4_t *carry_out, size_t vl) {
                *sum = __riscv_vadd_vv_u64m4(a, b, vl);
                vbool16_t carry_mask = __riscv_vmsltu_vv_u64m4_b16(*sum, a, vl);
                vuint64m4_t ones = __riscv_vmv_v_x_u64m4(1, vl);
                vuint64m4_t zeros = __riscv_vmv_v_x_u64m4(0, vl);
                *carry_out = __riscv_vmerge_vvm_u64m4(zeros, ones, carry_mask, vl);
            }
      
      
            void barrett_reduce_rvv(const uint64_t *operand1, const uint64_t *operand2, uint64_t *result,
                            uint64_t const_ratio_0, uint64_t const_ratio_1, uint64_t modulus_value, size_t coeff_count) {
                size_t i = 0;
                size_t vl = __riscv_vsetvl_e64m4(coeff_count);
            
                while (i < coeff_count) {
                    vuint64m4_t op1 = __riscv_vle64_v_u64m4(operand1 + i, vl);
                    vuint64m4_t op2 = __riscv_vle64_v_u64m4(operand2 + i, vl);
            
                    vuint64m4_t z_low = __riscv_vmul_vv_u64m4(op1, op2, vl);
                    vuint64m4_t z_high = __riscv_vmulhu_vv_u64m4(op1, op2, vl);
            
                    vuint64m4_t carry = __riscv_vmulhu_vx_u64m4(z_low, const_ratio_0, vl);
                    vuint64m4_t tmp2_lo = __riscv_vmul_vx_u64m4(z_low, const_ratio_1, vl);
                    vuint64m4_t tmp2_hi = __riscv_vmulhu_vx_u64m4(z_low, const_ratio_1, vl);
            
                    vuint64m4_t sum, carry2;
                    add_uint64_vector(tmp2_lo, carry, &sum, &carry2, vl);
            
                    vuint64m4_t tmp3 = __riscv_vadd_vv_u64m4(tmp2_hi, carry2, vl);
            
                    tmp2_lo = __riscv_vmul_vx_u64m4(z_high, const_ratio_0, vl);
                    tmp2_hi = __riscv_vmulhu_vx_u64m4(z_high, const_ratio_0, vl);
            
                    add_uint64_vector(tmp2_lo, sum, &sum, &carry2, vl);
                    carry = __riscv_vadd_vv_u64m4(carry2, tmp2_hi, vl);
            
                    carry2 = __riscv_vmul_vx_u64m4(z_high, const_ratio_1, vl);
                    carry2 = __riscv_vadd_vv_u64m4(carry2, tmp3, vl);
                    carry2 = __riscv_vadd_vv_u64m4(carry2, carry, vl);
            
                    vuint64m4_t estimate = __riscv_vmul_vx_u64m4(carry2, modulus_value, vl);
                    vuint64m4_t tmp3_final = __riscv_vsub_vv_u64m4(z_low, estimate, vl);
            
                    vbool16_t mask = __riscv_vmsgeu_vx_u64m4_b16(tmp3_final, modulus_value, vl);
                    vuint64m4_t modval_vec = __riscv_vmv_v_x_u64m4(modulus_value, vl);
                    vuint64m4_t corrected = __riscv_vsub_vv_u64m4(tmp3_final, modval_vec, vl);
            
                    vuint64m4_t final = __riscv_vmerge_vvm_u64m4(tmp3_final, corrected, mask, vl);
            
                    __riscv_vse64_v_u64m4(result + i, final, vl);
                    i += vl;
                }
            }
      #endif

        void modulo_poly_coeffs(ConstCoeffIter poly, std::size_t coeff_count, const Modulus &modulus, CoeffIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly && coeff_count > 0)
            {
                throw std::invalid_argument("poly");
            }
            if (!result && coeff_count > 0)
            {
                throw std::invalid_argument("result");
            }
            if (modulus.is_zero())
            {
                throw std::invalid_argument("modulus");
            }
#endif

#ifdef SEAL_USE_INTEL_HEXL
            intel::hexl::EltwiseReduceMod(result, poly, coeff_count, modulus.value(), modulus.value(), 1);
#else
            SEAL_ITERATE(
                iter(poly, result), coeff_count, [&](auto I) { get<1>(I) = barrett_reduce_64(get<0>(I), modulus); });
#endif
        }

        void add_poly_coeffmod(
            ConstCoeffIter operand1, ConstCoeffIter operand2, std::size_t coeff_count, const Modulus &modulus,
            CoeffIter result)
        {
#ifdef SEAL_DEBUG
            if (!operand1 && coeff_count > 0)
            {
                throw std::invalid_argument("operand1");
            }
            if (!operand2 && coeff_count > 0)
            {
                throw std::invalid_argument("operand2");
            }
            if (modulus.is_zero())
            {
                throw std::invalid_argument("modulus");
            }
            if (!result && coeff_count > 0)
            {
                throw std::invalid_argument("result");
            }
#endif
            const uint64_t modulus_value = modulus.value();

#ifdef SEAL_USE_INTEL_HEXL
            intel::hexl::EltwiseAddMod(&result[0], &operand1[0], &operand2[0], coeff_count, modulus_value);
#else

            SEAL_ITERATE(iter(operand1, operand2, result), coeff_count, [&](auto I) {
#ifdef SEAL_DEBUG
                if (get<0>(I) >= modulus_value)
                {
                    throw std::invalid_argument("operand1");
                }
                if (get<1>(I) >= modulus_value)
                {
                    throw std::invalid_argument("operand2");
                }
#endif
                std::uint64_t sum = get<0>(I) + get<1>(I);
                get<2>(I) = SEAL_COND_SELECT(sum >= modulus_value, sum - modulus_value, sum);
            });
#endif
        }

        void sub_poly_coeffmod(
            ConstCoeffIter operand1, ConstCoeffIter operand2, std::size_t coeff_count, const Modulus &modulus,
            CoeffIter result)
        {
#ifdef SEAL_DEBUG
            if (!operand1 && coeff_count > 0)
            {
                throw std::invalid_argument("operand1");
            }
            if (!operand2 && coeff_count > 0)
            {
                throw std::invalid_argument("operand2");
            }
            if (modulus.is_zero())
            {
                throw std::invalid_argument("modulus");
            }
            if (!result && coeff_count > 0)
            {
                throw std::invalid_argument("result");
            }
#endif

            const uint64_t modulus_value = modulus.value();
#ifdef SEAL_USE_INTEL_HEXL
            intel::hexl::EltwiseSubMod(result, operand1, operand2, coeff_count, modulus_value);
#else
            SEAL_ITERATE(iter(operand1, operand2, result), coeff_count, [&](auto I) {
#ifdef SEAL_DEBUG
                if (get<0>(I) >= modulus_value)
                {
                    throw std::invalid_argument("operand1");
                }
                if (get<1>(I) >= modulus_value)
                {
                    throw std::invalid_argument("operand2");
                }
#endif
                unsigned long long temp_result;
                std::int64_t borrow = sub_uint64(get<0>(I), get<1>(I), &temp_result);
                get<2>(I) = temp_result + (modulus_value & static_cast<std::uint64_t>(-borrow));
            });
#endif
        }

        void add_poly_scalar_coeffmod(
            ConstCoeffIter poly, size_t coeff_count, uint64_t scalar, const Modulus &modulus, CoeffIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly && coeff_count > 0)
            {
                throw invalid_argument("poly");
            }
            if (!result && coeff_count > 0)
            {
                throw invalid_argument("result");
            }
            if (modulus.is_zero())
            {
                throw invalid_argument("modulus");
            }
            if (scalar >= modulus.value())
            {
                throw invalid_argument("scalar");
            }
#endif

#ifdef SEAL_USE_INTEL_HEXL
            intel::hexl::EltwiseAddMod(result, poly, scalar, coeff_count, modulus.value());
#else
            SEAL_ITERATE(iter(poly, result), coeff_count, [&](auto I) {
                const uint64_t x = get<0>(I);
                get<1>(I) = add_uint_mod(x, scalar, modulus);
            });
#endif
        }

        void sub_poly_scalar_coeffmod(
            ConstCoeffIter poly, size_t coeff_count, uint64_t scalar, const Modulus &modulus, CoeffIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly && coeff_count > 0)
            {
                throw invalid_argument("poly");
            }
            if (!result && coeff_count > 0)
            {
                throw invalid_argument("result");
            }
            if (modulus.is_zero())
            {
                throw invalid_argument("modulus");
            }
            if (scalar >= modulus.value())
            {
                throw invalid_argument("scalar");
            }
#endif

#ifdef SEAL_USE_INTEL_HEXL
            intel::hexl::EltwiseSubMod(result, poly, scalar, coeff_count, modulus.value());
#else
            SEAL_ITERATE(iter(poly, result), coeff_count, [&](auto I) {
                const uint64_t x = get<0>(I);
                get<1>(I) = sub_uint_mod(x, scalar, modulus);
            });
#endif
        }

        void multiply_poly_scalar_coeffmod(
            ConstCoeffIter poly, size_t coeff_count, MultiplyUIntModOperand scalar, const Modulus &modulus,
            CoeffIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly && coeff_count > 0)
            {
                throw invalid_argument("poly");
            }
            if (!result && coeff_count > 0)
            {
                throw invalid_argument("result");
            }
            if (modulus.is_zero())
            {
                throw invalid_argument("modulus");
            }
#endif

#ifdef SEAL_USE_INTEL_HEXL
            intel::hexl::EltwiseFMAMod(&result[0], &poly[0], scalar.operand, nullptr, coeff_count, modulus.value(), 8);
#else
            SEAL_ITERATE(iter(poly, result), coeff_count, [&](auto I) {
                const uint64_t x = get<0>(I);
                get<1>(I) = multiply_uint_mod(x, scalar, modulus);
            });
#endif
        }

        void dyadic_product_coeffmod(
            ConstCoeffIter operand1, ConstCoeffIter operand2, size_t coeff_count, const Modulus &modulus,
            CoeffIter result)
        {
#ifdef SEAL_DEBUG
            if (!operand1)
            {
                throw invalid_argument("operand1");
            }
            if (!operand2)
            {
                throw invalid_argument("operand2");
            }
            if (!result)
            {
                throw invalid_argument("result");
            }
            if (coeff_count == 0)
            {
                throw invalid_argument("coeff_count");
            }
            if (modulus.is_zero())
            {
                throw invalid_argument("modulus");
            }
#endif
#ifdef SEAL_USE_INTEL_HEXL
            intel::hexl::EltwiseMultMod(&result[0], &operand1[0], &operand2[0], coeff_count, modulus.value(), 4);
#else
            const uint64_t modulus_value = modulus.value();
            const uint64_t const_ratio_0 = modulus.const_ratio()[0];
            const uint64_t const_ratio_1 = modulus.const_ratio()[1];
            #if defined(__riscv_v_intrinsic)
            
              barrett_reduce_rvv(operand1, operand2, result, const_ratio_0, const_ratio_1, modulus_value, coeff_count);
            
            
            #else

            SEAL_ITERATE(iter(operand1, operand2, result), coeff_count, [&](auto I) {
                // Reduces z using base 2^64 Barrett reduction
                unsigned long long z[2], tmp1, tmp2[2], tmp3, carry;
                multiply_uint64(get<0>(I), get<1>(I), z);

                // Multiply input and const_ratio
                // Round 1
                multiply_uint64_hw64(z[0], const_ratio_0, &carry);
                multiply_uint64(z[0], const_ratio_1, tmp2);
                tmp3 = tmp2[1] + add_uint64(tmp2[0], carry, &tmp1);

                // Round 2
                multiply_uint64(z[1], const_ratio_0, tmp2);
                carry = tmp2[1] + add_uint64(tmp1, tmp2[0], &tmp1);

                // This is all we care about
                tmp1 = z[1] * const_ratio_1 + tmp3 + carry;

                // Barrett subtraction
                tmp3 = z[0] - tmp1 * modulus_value;

                // Claim: One more subtraction is enough
                get<2>(I) = SEAL_COND_SELECT(tmp3 >= modulus_value, tmp3 - modulus_value, tmp3);
            });
            #endif
#endif
        }

        uint64_t poly_infty_norm_coeffmod(ConstCoeffIter operand, size_t coeff_count, const Modulus &modulus)
        {
#ifdef SEAL_DEBUG
            if (!operand && coeff_count > 0)
            {
                throw invalid_argument("operand");
            }
            if (modulus.is_zero())
            {
                throw invalid_argument("modulus");
            }
#endif
            // Construct negative threshold (first negative modulus value) to compute absolute values of coeffs.
            uint64_t modulus_neg_threshold = (modulus.value() + 1) >> 1;

            // Mod out the poly coefficients and choose a symmetric representative from
            // [-modulus,modulus). Keep track of the max.
            uint64_t result = 0;
            SEAL_ITERATE(operand, coeff_count, [&](auto I) {
                uint64_t poly_coeff = barrett_reduce_64(I, modulus);
                if (poly_coeff >= modulus_neg_threshold)
                {
                    poly_coeff = modulus.value() - poly_coeff;
                }
                if (poly_coeff > result)
                {
                    result = poly_coeff;
                }
            });

            return result;
        }

        void negacyclic_shift_poly_coeffmod(
            ConstCoeffIter poly, size_t coeff_count, size_t shift, const Modulus &modulus, CoeffIter result)
        {
#ifdef SEAL_DEBUG
            if (!poly)
            {
                throw invalid_argument("poly");
            }
            if (!result)
            {
                throw invalid_argument("result");
            }
            if (poly == result)
            {
                throw invalid_argument("result cannot point to the same value as poly");
            }
            if (modulus.is_zero())
            {
                throw invalid_argument("modulus");
            }
            if (util::get_power_of_two(static_cast<uint64_t>(coeff_count)) < 0)
            {
                throw invalid_argument("coeff_count");
            }
            if (shift >= coeff_count)
            {
                throw invalid_argument("shift");
            }
#endif
            // Nothing to do
            if (shift == 0)
            {
                set_uint(poly, coeff_count, result);
                return;
            }

            uint64_t index_raw = shift;
            uint64_t coeff_count_mod_mask = static_cast<uint64_t>(coeff_count) - 1;
            for (size_t i = 0; i < coeff_count; i++, poly++, index_raw++)
            {
                uint64_t index = index_raw & coeff_count_mod_mask;
                if (!(index_raw & static_cast<uint64_t>(coeff_count)) || !*poly)
                {
                    result[index] = *poly;
                }
                else
                {
                    result[index] = modulus.value() - *poly;
                }
            }
        }
    } // namespace util
} // namespace seal
