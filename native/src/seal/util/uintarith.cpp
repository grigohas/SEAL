// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "seal/util/common.h"
#include "seal/util/uintarith.h"
#include "seal/util/uintcore.h"
#include <algorithm>
#include <array>
#include <stdint.h>
#include <stdexcept>

#ifdef __riscv_v_intrinsic
#include <riscv_vector.h>
#endif

using namespace std;

namespace seal
{
    namespace util
    {
        void multiply_uint(
            const uint64_t *operand1, size_t operand1_uint64_count, const uint64_t *operand2,
            size_t operand2_uint64_count, size_t result_uint64_count, uint64_t *result)
        {
#ifdef SEAL_DEBUG
            if (!operand1 && operand1_uint64_count > 0)
            {
                throw invalid_argument("operand1");
            }
            if (!operand2 && operand2_uint64_count > 0)
            {
                throw invalid_argument("operand2");
            }
            if (!result_uint64_count)
            {
                throw invalid_argument("result_uint64_count");
            }
            if (!result)
            {
                throw invalid_argument("result");
            }
            if (result != nullptr && (operand1 == result || operand2 == result))
            {
                throw invalid_argument("result cannot point to the same value as operand1 or operand2");
            }
#endif
            // Handle fast cases.
            if (!operand1_uint64_count || !operand2_uint64_count)
            {
                // If either operand is 0, then result is 0.
                set_zero_uint(result_uint64_count, result);
                return;
            }
            if (result_uint64_count == 1)
            {
                *result = *operand1 * *operand2;
                return;
            }

            // In some cases these improve performance.
            operand1_uint64_count = get_significant_uint64_count_uint(operand1, operand1_uint64_count);
            operand2_uint64_count = get_significant_uint64_count_uint(operand2, operand2_uint64_count);

            // More fast cases
            if (operand1_uint64_count == 1)
            {
                multiply_uint(operand2, operand2_uint64_count, *operand1, result_uint64_count, result);
                return;
            }
            if (operand2_uint64_count == 1)
            {
                multiply_uint(operand1, operand1_uint64_count, *operand2, result_uint64_count, result);
                return;
            }

            // Clear out result.
            set_zero_uint(result_uint64_count, result);

            // Multiply operand1 and operand2.
            size_t operand1_index_max = min(operand1_uint64_count, result_uint64_count);
            for (size_t operand1_index = 0; operand1_index < operand1_index_max; operand1_index++)
            {
                const uint64_t *inner_operand2 = operand2;
                uint64_t *inner_result = result++;
                uint64_t carry = 0;
                size_t operand2_index = 0;
                size_t operand2_index_max = min(operand2_uint64_count, result_uint64_count - operand1_index);
                for (; operand2_index < operand2_index_max; operand2_index++)
                {
                    // Perform 64-bit multiplication of operand1 and operand2
                    unsigned long long temp_result[2];
                    multiply_uint64(*operand1, *inner_operand2++, temp_result);
                    carry = temp_result[1] + add_uint64(temp_result[0], carry, 0, temp_result);
                    unsigned long long temp;
                    carry += add_uint64(*inner_result, temp_result[0], 0, &temp);
                    *inner_result++ = temp;
                }

                // Write carry if there is room in result
                if (operand1_index + operand2_index_max < result_uint64_count)
                {
                    *inner_result = carry;
                }

                operand1++;
            }
        }

        void multiply_uint(
            const uint64_t *operand1, size_t operand1_uint64_count, uint64_t operand2, size_t result_uint64_count,
            uint64_t *result)
        {
#ifdef SEAL_DEBUG
            if (!operand1 && operand1_uint64_count > 0)
            {
                throw invalid_argument("operand1");
            }
            if (!result_uint64_count)
            {
                throw invalid_argument("result_uint64_count");
            }
            if (!result)
            {
                throw invalid_argument("result");
            }
            if (result != nullptr && operand1 == result)
            {
                throw invalid_argument("result cannot point to the same value as operand1");
            }
#endif
            // Handle fast cases.
            if (!operand1_uint64_count || !operand2)
            {
                // If either operand is 0, then result is 0.
                set_zero_uint(result_uint64_count, result);
                return;
            }
            if (result_uint64_count == 1)
            {
                *result = *operand1 * operand2;
                return;
            }

            // Clear out result.
            set_zero_uint(result_uint64_count, result);

            // Multiply operand1 and operand2.
            unsigned long long carry = 0;
            size_t operand1_index_max = min(operand1_uint64_count, result_uint64_count);
            for (size_t operand1_index = 0; operand1_index < operand1_index_max; operand1_index++)
            {
                unsigned long long temp_result[2];
                multiply_uint64(*operand1++, operand2, temp_result);
                unsigned long long temp;
                carry = temp_result[1] + add_uint64(temp_result[0], carry, 0, &temp);
                *result++ = temp;
            }

            // Write carry if there is room in result
            if (operand1_index_max < result_uint64_count)
            {
                *result = carry;
            }
        }

        void divide_uint_inplace(
            uint64_t *numerator, const uint64_t *denominator, size_t uint64_count, uint64_t *quotient, MemoryPool &pool)
        {
#ifdef SEAL_DEBUG
            if (!numerator && uint64_count > 0)
            {
                throw invalid_argument("numerator");
            }
            if (!denominator && uint64_count > 0)
            {
                throw invalid_argument("denominator");
            }
            if (!quotient && uint64_count > 0)
            {
                throw invalid_argument("quotient");
            }
            if (is_zero_uint(denominator, uint64_count) && uint64_count > 0)
            {
                throw invalid_argument("denominator");
            }
            if (quotient && (numerator == quotient || denominator == quotient))
            {
                throw invalid_argument("quotient cannot point to same value as numerator or denominator");
            }
#endif
            if (!uint64_count)
            {
                return;
            }

            // Clear quotient. Set it to zero.
            set_zero_uint(uint64_count, quotient);

            // Determine significant bits in numerator and denominator.
            int numerator_bits = get_significant_bit_count_uint(numerator, uint64_count);
            int denominator_bits = get_significant_bit_count_uint(denominator, uint64_count);

            // If numerator has fewer bits than denominator, then done.
            if (numerator_bits < denominator_bits)
            {
                return;
            }

            // Only perform computation up to last non-zero uint64s.
            uint64_count = safe_cast<size_t>(divide_round_up(numerator_bits, bits_per_uint64));

            // Handle fast case.
            if (uint64_count == 1)
            {
                *quotient = *numerator / *denominator;
                *numerator -= *quotient * *denominator;
                return;
            }

            auto alloc_anchor(allocate_uint(uint64_count << 1, pool));

            // Create temporary space to store mutable copy of denominator.
            uint64_t *shifted_denominator = alloc_anchor.get();

            // Create temporary space to store difference calculation.
            uint64_t *difference = shifted_denominator + uint64_count;

            // Shift denominator to bring MSB in alignment with MSB of numerator.
            int denominator_shift = numerator_bits - denominator_bits;
            left_shift_uint(denominator, denominator_shift, uint64_count, shifted_denominator);
            denominator_bits += denominator_shift;

            // Perform bit-wise division algorithm.
            int remaining_shifts = denominator_shift;
            while (numerator_bits == denominator_bits)
            {
                // NOTE: MSBs of numerator and denominator are aligned.

                // Even though MSB of numerator and denominator are aligned,
                // still possible numerator < shifted_denominator.
                if (sub_uint(numerator, shifted_denominator, uint64_count, difference))
                {
                    // numerator < shifted_denominator and MSBs are aligned,
                    // so current quotient bit is zero and next one is definitely one.
                    if (remaining_shifts == 0)
                    {
                        // No shifts remain and numerator < denominator so done.
                        break;
                    }

                    // Effectively shift numerator left by 1 by instead adding
                    // numerator to difference (to prevent overflow in numerator).
                    add_uint(difference, numerator, uint64_count, difference);

                    // Adjust quotient and remaining shifts as a result of
                    // shifting numerator.
                    left_shift_uint(quotient, 1, uint64_count, quotient);
                    remaining_shifts--;
                }
                // Difference is the new numerator with denominator subtracted.

                // Update quotient to reflect subtraction.
                quotient[0] |= 1;

                // Determine amount to shift numerator to bring MSB in alignment
                // with denominator.
                numerator_bits = get_significant_bit_count_uint(difference, uint64_count);
                int numerator_shift = denominator_bits - numerator_bits;
                if (numerator_shift > remaining_shifts)
                {
                    // Clip the maximum shift to determine only the integer
                    // (as opposed to fractional) bits.
                    numerator_shift = remaining_shifts;
                }

                // Shift and update numerator.
                if (numerator_bits > 0)
                {
                    left_shift_uint(difference, numerator_shift, uint64_count, numerator);
                    numerator_bits += numerator_shift;
                }
                else
                {
                    // Difference is zero so no need to shift, just set to zero.
                    set_zero_uint(uint64_count, numerator);
                }

                // Adjust quotient and remaining shifts as a result of shifting numerator.
                left_shift_uint(quotient, numerator_shift, uint64_count, quotient);
                remaining_shifts -= numerator_shift;
            }

            // Correct numerator (which is also the remainder) for shifting of
            // denominator, unless it is just zero.
            if (numerator_bits > 0)
            {
                right_shift_uint(numerator, denominator_shift, uint64_count, numerator);
            }
        }


        void divide_uint128_uint64_inplace_generic(uint64_t *numerator, uint64_t denominator, uint64_t *quotient)
        {
        #ifdef SEAL_DEBUG
            if (!numerator)
            {
                throw std::invalid_argument("numerator");
            }
            if (denominator == 0)
            {
                throw std::invalid_argument("denominator");
            }
            if (!quotient)
            {
                throw std::invalid_argument("quotient");
            }
            if (numerator == quotient)
            {
                throw std::invalid_argument("quotient cannot point to the same value as numerator");
            }
        #endif
        
        #if defined __riscv_v_intrinsic
            constexpr size_t uint64_count = 2;

            // Initialize quotient to zero
            quotient[0] = 0;
            quotient[1] = 0;
        
            // Set vector length for ELEN=64, VLEN=256 (max 4 elements)
            size_t vl = vsetvlmax_e64m1(); 
        
            // Load numerator and denominator into vector registers
            vuint64m1_t num_vec = vle64_v_u64m1(numerator, vl);
            vuint64m1_t den_vec = vfmv_v_f_u64m1(denominator, vl); // Broadcast denominator
        
            // Get significant bits
            int numerator_bits = get_significant_bit_count_uint(numerator, uint64_count);
            int denominator_bits = get_significant_bit_count(denominator);
        
            if (numerator_bits < denominator_bits)
            {
                return;
            }
        
            // Create temporary storage for the shifted denominator
            uint64_t temp_denominator[uint64_count] = { denominator, 0 };
            vuint64m1_t shifted_denominator = vle64_v_u64m1(temp_denominator, vl);
            vuint64m1_t difference = vmv_v_x_u64m1(0, vl); // Initialize difference to zero
        
            int denominator_shift = numerator_bits - denominator_bits;
        
            // Left shift denominator to align with numerator
            shifted_denominator = vsll_vx_u64m1(shifted_denominator, denominator_shift, vl);
            denominator_bits += denominator_shift;
        
            // Perform bitwise division algorithm
            int remaining_shifts = denominator_shift;
            while (numerator_bits == denominator_bits)
            {
                // Subtract numerator - shifted_denominator
                vuint64m1_t diff_vec = vsub_vv_u64m1(num_vec, shifted_denominator, vl);
                
                // Check if subtraction resulted in a borrow
                vbool64_t borrow_mask = vmsltu_vv_u64m1_b64(num_vec, shifted_denominator, vl);
        
                if (vfirst_b64(borrow_mask, vl) != -1) // If borrow occurred
                {
                    if (remaining_shifts == 0)
                    {
                        break;
                    }
        
                    // Add back the difference to prevent underflow
                    num_vec = vadd_vv_u64m1(diff_vec, num_vec, vl);
        
                    // Shift quotient left
                    uint64_t carry = quotient[0] >> 63;  // Carry bit from quotient[0] to quotient[1]
                    quotient[1] = (quotient[1] << 1) | carry;
                    quotient[0] <<= 1;
        
                    remaining_shifts--;
                }
        
                // Update numerator
                numerator_bits = get_significant_bit_count_uint((uint64_t*)&num_vec, uint64_count);
        
                // Determine shift amount
                int numerator_shift = MIN(denominator_bits - numerator_bits, remaining_shifts);
        
                if (numerator_bits > 0)
                {
                    num_vec = vsll_vx_u64m1(diff_vec, numerator_shift, vl);
                    numerator_bits += numerator_shift;
                }
        
                // Update quotient
                quotient[0] |= 1;
                uint64_t carry = quotient[0] >> 63;
                quotient[1] = (quotient[1] << numerator_shift) | carry;
                quotient[0] <<= numerator_shift;
        
                remaining_shifts -= numerator_shift;
            }
        
            // Store final remainder (numerator)
            vse64_v_u64m1(numerator, vsrl_vx_u64m1(num_vec, denominator_shift, vl), vl)
        
        #else
            constexpr size_t uint64_count = 2;
        
            // Clear quotient
            quotient[0] = 0;
            quotient[1] = 0;
        
            // Get significant bits
            int numerator_bits = get_significant_bit_count_uint(numerator, uint64_count);
            int denominator_bits = get_significant_bit_count(denominator);
        
            if (numerator_bits < denominator_bits)
            {
                return;
            }
        
            // Create storage for shifted denominator
            uint64_t shifted_denominator[uint64_count]{ denominator, 0 };
            uint64_t difference[uint64_count]{ 0, 0 };
        
            // Shift denominator
            int denominator_shift = numerator_bits - denominator_bits;
            left_shift_uint128(shifted_denominator, denominator_shift, shifted_denominator);
            denominator_bits += denominator_shift;
        
            int remaining_shifts = denominator_shift;
            while (numerator_bits == denominator_bits)
            {
                if (sub_uint(numerator, shifted_denominator, uint64_count, difference))
                {
                    if (remaining_shifts == 0)
                    {
                        break;
                    }
        
                    add_uint(difference, numerator, uint64_count, difference);
        
                    quotient[1] = (quotient[1] << 1) | (quotient[0] >> (63));
                    quotient[0] <<= 1;
                    remaining_shifts--;
                }
        
                numerator_bits = get_significant_bit_count_uint(difference, uint64_count);
        
                int numerator_shift = std::min(denominator_bits - numerator_bits, remaining_shifts);
        
                numerator[0] = 0;
                numerator[1] = 0;
        
                if (numerator_bits > 0)
                {
                    left_shift_uint128(difference, numerator_shift, numerator);
                    numerator_bits += numerator_shift;
                }
        
                quotient[0] |= 1;
        
                left_shift_uint128(quotient, numerator_shift, quotient);
                remaining_shifts -= numerator_shift;
            }
        
            if (numerator_bits > 0)
            {
                right_shift_uint128(numerator, denominator_shift, numerator);
            }
        #endif
        }


        void divide_uint192_inplace(uint64_t *numerator, uint64_t denominator, uint64_t *quotient)
        {
#ifdef SEAL_DEBUG
            if (!numerator)
            {
                throw invalid_argument("numerator");
            }
            if (denominator == 0)
            {
                throw invalid_argument("denominator");
            }
            if (!quotient)
            {
                throw invalid_argument("quotient");
            }
            if (numerator == quotient)
            {
                throw invalid_argument("quotient cannot point to same value as numerator");
            }
#endif
            // We expect 192-bit input
            size_t uint64_count = 3;

            // Clear quotient. Set it to zero.
            quotient[0] = 0;
            quotient[1] = 0;
            quotient[2] = 0;

            // Determine significant bits in numerator and denominator.
            int numerator_bits = get_significant_bit_count_uint(numerator, uint64_count);
            int denominator_bits = get_significant_bit_count(denominator);

            // If numerator has fewer bits than denominator, then done.
            if (numerator_bits < denominator_bits)
            {
                return;
            }

            // Only perform computation up to last non-zero uint64s.
            uint64_count = safe_cast<size_t>(divide_round_up(numerator_bits, bits_per_uint64));

            // Handle fast case.
            if (uint64_count == 1)
            {
                *quotient = *numerator / denominator;
                *numerator -= *quotient * denominator;
                return;
            }

            // Create temporary space to store mutable copy of denominator.
            vector<uint64_t> shifted_denominator(uint64_count, 0);
            shifted_denominator[0] = denominator;

            // Create temporary space to store difference calculation.
            vector<uint64_t> difference(uint64_count);

            // Shift denominator to bring MSB in alignment with MSB of numerator.
            int denominator_shift = numerator_bits - denominator_bits;

            left_shift_uint192(shifted_denominator.data(), denominator_shift, shifted_denominator.data());
            denominator_bits += denominator_shift;

            // Perform bit-wise division algorithm.
            int remaining_shifts = denominator_shift;
            while (numerator_bits == denominator_bits)
            {
                // NOTE: MSBs of numerator and denominator are aligned.

                // Even though MSB of numerator and denominator are aligned,
                // still possible numerator < shifted_denominator.
                if (sub_uint(numerator, shifted_denominator.data(), uint64_count, difference.data()))
                {
                    // numerator < shifted_denominator and MSBs are aligned,
                    // so current quotient bit is zero and next one is definitely one.
                    if (remaining_shifts == 0)
                    {
                        // No shifts remain and numerator < denominator so done.
                        break;
                    }

                    // Effectively shift numerator left by 1 by instead adding
                    // numerator to difference (to prevent overflow in numerator).
                    add_uint(difference.data(), numerator, uint64_count, difference.data());

                    // Adjust quotient and remaining shifts as a result of shifting numerator.
                    left_shift_uint192(quotient, 1, quotient);
                    remaining_shifts--;
                }
                // Difference is the new numerator with denominator subtracted.

                // Update quotient to reflect subtraction.
                quotient[0] |= 1;

                // Determine amount to shift numerator to bring MSB in alignment with denominator.
                numerator_bits = get_significant_bit_count_uint(difference.data(), uint64_count);
                int numerator_shift = denominator_bits - numerator_bits;
                if (numerator_shift > remaining_shifts)
                {
                    // Clip the maximum shift to determine only the integer
                    // (as opposed to fractional) bits.
                    numerator_shift = remaining_shifts;
                }

                // Shift and update numerator.
                if (numerator_bits > 0)
                {
                    left_shift_uint192(difference.data(), numerator_shift, numerator);
                    numerator_bits += numerator_shift;
                }
                else
                {
                    // Difference is zero so no need to shift, just set to zero.
                    set_zero_uint(uint64_count, numerator);
                }

                // Adjust quotient and remaining shifts as a result of shifting numerator.
                left_shift_uint192(quotient, numerator_shift, quotient);
                remaining_shifts -= numerator_shift;
            }

            // Correct numerator (which is also the remainder) for shifting of
            // denominator, unless it is just zero.
            if (numerator_bits > 0)
            {
                right_shift_uint192(numerator, denominator_shift, numerator);
            }
        }

        uint64_t exponentiate_uint_safe(uint64_t operand, uint64_t exponent)
        {
            // Fast cases
            if (exponent == 0)
            {
                return 1;
            }
            if (exponent == 1)
            {
                return operand;
            }

            // Perform binary exponentiation.
            uint64_t power = operand;
            uint64_t product = 0;
            uint64_t intermediate = 1;

            // Initially: power = operand and intermediate = 1, product irrelevant.
            while (true)
            {
                if (exponent & 1)
                {
                    product = mul_safe(power, intermediate);
                    swap(product, intermediate);
                }
                exponent >>= 1;
                if (exponent == 0)
                {
                    break;
                }
                product = mul_safe(power, power);
                swap(product, power);
            }

            return intermediate;
        }

        uint64_t exponentiate_uint(uint64_t operand, uint64_t exponent)
        {
            // Fast cases
            if (exponent == 0)
            {
                return 1;
            }
            if (exponent == 1)
            {
                return operand;
            }

            // Perform binary exponentiation.
            uint64_t power = operand;
            uint64_t product = 0;
            uint64_t intermediate = 1;

            // Initially: power = operand and intermediate = 1, product irrelevant.
            while (true)
            {
                if (exponent & 1)
                {
                    product = power * intermediate;
                    swap(product, intermediate);
                }
                exponent >>= 1;
                if (exponent == 0)
                {
                    break;
                }
                product = power * power;
                swap(product, power);
            }

            return intermediate;
        }
    } // namespace util
} // namespace seal
