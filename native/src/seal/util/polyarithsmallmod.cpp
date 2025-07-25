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
using namespace std::chrono;

extern long int q;

using namespace std;

namespace seal
{
    namespace util
    {
        #if defined(__riscv_v_intrinsic)
        
             inline vuint64m4_t dyadic_product_coeffmod_rvv(vuint64m4_t op1, vuint64m4_t op2,uint64_t const_ratio_0, uint64_t const_ratio_1, uint64_t modulus_value,size_t vl) {
                
                // Step 1: 128-bit multiplication op1 * op2
                vuint64m4_t z_low = __riscv_vmul_vv_u64m4(op1, op2, vl);
                vuint64m4_t z_high = __riscv_vmulhu_vv_u64m4(op1, op2, vl);
                
                // Step 2: Use scalar-vector operations (more efficient)
                vuint64m4_t carry1 = __riscv_vmulhu_vx_u64m4(z_low, const_ratio_0, vl);
                vuint64m4_t tmp2_lo = __riscv_vmul_vx_u64m4(z_low, const_ratio_1, vl);
                vuint64m4_t tmp2_hi = __riscv_vmulhu_vx_u64m4(z_low, const_ratio_1, vl);
                
                vuint64m4_t sum1 = __riscv_vadd_vv_u64m4(tmp2_lo, carry1, vl);
                vbool16_t overflow1 = __riscv_vmsltu_vv_u64m4_b16(sum1, tmp2_lo, vl);
                
                vuint64m4_t high1 = __riscv_vadd_vv_u64m4(tmp2_hi,__riscv_vmerge_vxm_u64m4(__riscv_vmv_v_x_u64m4(0, vl), 1, overflow1, vl), vl);
                
                vuint64m4_t tmp3_lo = __riscv_vmul_vx_u64m4(z_high, const_ratio_0, vl);
                vuint64m4_t tmp3_hi = __riscv_vmulhu_vx_u64m4(z_high, const_ratio_0, vl);
                
                vuint64m4_t sum2 = __riscv_vadd_vv_u64m4(sum1, tmp3_lo, vl);
                vbool16_t overflow2 = __riscv_vmsltu_vv_u64m4_b16(sum2, sum1, vl);
                
                vuint64m4_t high2 = __riscv_vadd_vv_u64m4(high1, tmp3_hi, vl);
                high2 = __riscv_vadd_vv_u64m4(high2,__riscv_vmerge_vxm_u64m4(__riscv_vmv_v_x_u64m4(0, vl), 1, overflow2, vl), vl);
                
                vuint64m4_t tmp4 = __riscv_vmul_vx_u64m4(z_high, const_ratio_1, vl);
                vuint64m4_t quotient = __riscv_vadd_vv_u64m4(high2, tmp4, vl);
                
                // Step 3: Use scalar-vector operations for final steps
                vuint64m4_t estimate = __riscv_vmul_vx_u64m4(quotient, modulus_value, vl);
                vuint64m4_t remainder = __riscv_vsub_vv_u64m4(z_low, estimate, vl);
                
                vbool16_t needs_correction = __riscv_vmsgeu_vx_u64m4_b16(remainder, modulus_value, vl);
                vuint64m4_t corrected = __riscv_vsub_vx_u64m4(remainder, modulus_value, vl);
                
                return __riscv_vmerge_vvm_u64m4(remainder, corrected, needs_correction, vl);
            }

            inline vuint64m4_t multiply_uint_mod_rvv(const vuint64m4_t a, const uint64_t yquot,const uint64_t yop, const Modulus &modulus, size_t vl) {
                vuint64m4_t vp = __riscv_vmv_v_x_u64m4(modulus.value(), vl);
                
                vuint64m4_t vhi = __riscv_vmulhu_vx_u64m4(a, yquot, vl);
                vuint64m4_t vmul1 = __riscv_vmul_vx_u64m4(a, yop, vl);
                vuint64m4_t vmul2 = __riscv_vmul_vv_u64m4(vhi, vp, vl);
                vuint64m4_t vres = __riscv_vsub_vv_u64m4(vmul1, vmul2, vl);
                
                vbool16_t ge_mask = __riscv_vmsgeu_vv_u64m4_b16(vres, vp, vl);
                vuint64m4_t vcorrected = __riscv_vsub_vv_u64m4(vres, vp, vl);
                return __riscv_vmerge_vvm_u64m4(vres, vcorrected, ge_mask, vl);
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

            #if defined(__riscv_v_intrinsic)
                 size_t processed=0;
                 
                 while (processed < coeff_count) {
                    size_t vl = __riscv_vsetvl_e64m4(coeff_count - processed);
                    vuint64m4_t vx = __riscv_vle64_v_u64m4(poly + processed, vl);
                    vuint64m4_t vv = multiply_uint_mod_rvv(vx,scalar.quotient, scalar.operand, modulus,vl) ;
                    
                    __riscv_vse64_v_u64m4(result + processed, vv, vl);
                    processed += vl;
                }
            #else
            SEAL_ITERATE(iter(poly, result), coeff_count, [&](auto I) {
                const uint64_t x = get<0>(I);
                get<1>(I) = multiply_uint_mod(x, scalar, modulus);
            });
            #endif
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
            auto start4 = high_resolution_clock::now();
            #if defined(__riscv_v_intrinsic)  
            size_t processed = 0;
            
            while (processed < coeff_count) {
                size_t vl = __riscv_vsetvl_e64m4(coeff_count - processed);
                
                vuint64m4_t vop1 = __riscv_vle64_v_u64m4(operand1 + processed, vl);
                vuint64m4_t vop2 = __riscv_vle64_v_u64m4(operand2 + processed, vl);
                
                // Use scalar constants - NO vector creation needed!
                vuint64m4_t vres = dyadic_product_coeffmod_rvv(vop1, vop2,const_ratio_0, const_ratio_1, modulus_value, vl);
                
                __riscv_vse64_v_u64m4(result + processed, vres, vl);
                processed += vl;
            }
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
            auto stop4 = high_resolution_clock::now();
   	        auto duration4 = duration_cast<microseconds>(stop4 - start4);
            q+=duration4.count();
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
