// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "seal/memorymanager.h"
#include "seal/modulus.h"
#include "seal/util/defines.h"
#include "seal/util/dwthandler.h"
#include "seal/util/iterator.h"
#include "seal/util/pointer.h"
#include "seal/util/uintarithsmallmod.h"
#include "seal/util/uintcore.h"
#include <stdexcept>

#ifdef __riscv_vector
#include <riscv_vector.h>
#endif


namespace seal
{
    namespace util
    {
        template <>
        class Arithmetic<std::uint64_t, MultiplyUIntModOperand, MultiplyUIntModOperand>
        {
        public:
            Arithmetic()
            {}

            Arithmetic(const Modulus &modulus) : modulus_(modulus), two_times_modulus_(modulus.value() << 1)
            {}

            inline std::uint64_t add(const std::uint64_t &a, const std::uint64_t &b) const
            {
                return a + b;
            }

            inline std::uint64_t sub(const std::uint64_t &a, const std::uint64_t &b) const
            {
                return a + two_times_modulus_ - b;
            }

            inline std::uint64_t mul_root(const std::uint64_t &a, const MultiplyUIntModOperand &r) const
            {
                return multiply_uint_mod_lazy(a, r, modulus_);
            }

            inline std::uint64_t mul_scalar(const std::uint64_t &a, const MultiplyUIntModOperand &s) const
            {
                return multiply_uint_mod_lazy(a, s, modulus_);
            }

            inline MultiplyUIntModOperand mul_root_scalar(
                const MultiplyUIntModOperand &r, const MultiplyUIntModOperand &s) const
            {
                MultiplyUIntModOperand result;
                result.set(multiply_uint_mod(r.operand, s, modulus_), modulus_);
                return result;
            }

            inline std::uint64_t guard(const std::uint64_t &a) const
            {
                return SEAL_COND_SELECT(a >= two_times_modulus_, a - two_times_modulus_, a);
            }

            #if defined(__riscv_v_intrinsic)
            
                 inline vuint64m4_t guard_vector_rvv(const vuint64m4_t in, size_t vl) const {
                      vuint64m4_t modulus_vec = __riscv_vmv_v_x_u64m4(two_times_modulus_, vl);
                      vbool16_t mask = __riscv_vmsgeu_vx_u64m4_b16(in, two_times_modulus_, vl);
                      return __riscv_vsub_vv_u64m4_mu(mask, in, in, modulus_vec, vl);
                  }
                  
                  inline vuint64m4_t add_vector_rvv(const vuint64m4_t a, const vuint64m4_t b, size_t vl) const {
                      return __riscv_vadd_vv_u64m4(a, b, vl);
                  }
                  
                  inline vuint64m4_t sub_vector_rvv(const vuint64m4_t a, const vuint64m4_t b, size_t vl) const {
                      // Broadcast vector with all elements = two_times_modulus_
                      vuint64m4_t modulus_vec = __riscv_vmv_v_x_u64m4(two_times_modulus_, vl);
                      // result = a + (2 * modulus) - b
                      vuint64m4_t tmp = __riscv_vadd_vv_u64m4(a, modulus_vec, vl);
                      vuint64m4_t result = __riscv_vsub_vv_u64m4(tmp, b, vl);
                      return result;
                  }
                  
                  inline vuint64m4_t mul_vector_rvv(const vuint64m4_t a, const uint64_t yquot, const uint64_t yop, size_t vl) const {
                      const uint64_t p = modulus_.value();  // Assuming modulus_ is in scope
                  
                      // Replicate scalars across vector registers
                      vuint64m4_t vb = __riscv_vmv_v_x_u64m4(yquot, vl);
                      vuint64m4_t vp = __riscv_vmv_v_x_u64m4(p, vl);
                      vuint64m4_t vop = __riscv_vmv_v_x_u64m4(yop, vl);
                  
                      // Unsigned high part of a * yquot
                      vuint64m4_t vhi = __riscv_vmulhu_vv_u64m4(a, vb, vl);
                  
                      // a * yop
                      vuint64m4_t vmul1 = __riscv_vmul_vv_u64m4(a, vop, vl);
                  
                      // vhi * p
                      vuint64m4_t vmul2 = __riscv_vmul_vv_u64m4(vhi, vp, vl);
                  
                      // (a * yop) - (vhi * p)
                      vuint64m4_t vres = __riscv_vsub_vv_u64m4(vmul1, vmul2, vl);
                  
                      return vres;
                  }
              #endif

        private:
            Modulus modulus_;

            std::uint64_t two_times_modulus_;
        };

        class NTTTables
        {
            using ModArithLazy = Arithmetic<uint64_t, MultiplyUIntModOperand, MultiplyUIntModOperand>;
            using NTTHandler = DWTHandler<std::uint64_t, MultiplyUIntModOperand, MultiplyUIntModOperand>;

        public:
            NTTTables(NTTTables &&source) = default;

            NTTTables(NTTTables &copy)
                : pool_(copy.pool_), root_(copy.root_), coeff_count_power_(copy.coeff_count_power_),
                  coeff_count_(copy.coeff_count_), modulus_(copy.modulus_), inv_degree_modulo_(copy.inv_degree_modulo_)
            {
                root_powers_ = allocate<MultiplyUIntModOperand>(coeff_count_, pool_);
                inv_root_powers_ = allocate<MultiplyUIntModOperand>(coeff_count_, pool_);

                std::copy_n(copy.root_powers_.get(), coeff_count_, root_powers_.get());
                std::copy_n(copy.inv_root_powers_.get(), coeff_count_, inv_root_powers_.get());
            }

            NTTTables(int coeff_count_power, const Modulus &modulus, MemoryPoolHandle pool = MemoryManager::GetPool());

            SEAL_NODISCARD inline std::uint64_t get_root() const
            {
                return root_;
            }

            SEAL_NODISCARD inline const MultiplyUIntModOperand *get_from_root_powers() const
            {
                return root_powers_.get();
            }

            SEAL_NODISCARD inline const MultiplyUIntModOperand *get_from_inv_root_powers() const
            {
                return inv_root_powers_.get();
            }

            SEAL_NODISCARD inline MultiplyUIntModOperand get_from_root_powers(std::size_t index) const
            {
#ifdef SEAL_DEBUG
                if (index >= coeff_count_)
                {
                    throw std::out_of_range("index");
                }
#endif
                return root_powers_[index];
            }

            SEAL_NODISCARD inline MultiplyUIntModOperand get_from_inv_root_powers(std::size_t index) const
            {
#ifdef SEAL_DEBUG
                if (index >= coeff_count_)
                {
                    throw std::out_of_range("index");
                }
#endif
                return inv_root_powers_[index];
            }

            SEAL_NODISCARD inline const MultiplyUIntModOperand &inv_degree_modulo() const
            {
                return inv_degree_modulo_;
            }

            SEAL_NODISCARD inline const Modulus &modulus() const
            {
                return modulus_;
            }

            SEAL_NODISCARD inline int coeff_count_power() const
            {
                return coeff_count_power_;
            }

            SEAL_NODISCARD inline std::size_t coeff_count() const
            {
                return coeff_count_;
            }

            const NTTHandler &ntt_handler() const
            {
                return ntt_handler_;
            }

        private:
            NTTTables &operator=(const NTTTables &assign) = delete;

            NTTTables &operator=(NTTTables &&assign) = delete;

            void initialize(int coeff_count_power, const Modulus &modulus);

            MemoryPoolHandle pool_;

            std::uint64_t root_ = 0;

            std::uint64_t inv_root_ = 0;

            int coeff_count_power_ = 0;

            std::size_t coeff_count_ = 0;

            Modulus modulus_;

            // Inverse of coeff_count_ modulo modulus_.
            MultiplyUIntModOperand inv_degree_modulo_;

            // Holds 1~(n-1)-th powers of root_ in bit-reversed order, the 0-th power is left unset.
            Pointer<MultiplyUIntModOperand> root_powers_;

            // Holds 1~(n-1)-th powers of inv_root_ in scrambled order, the 0-th power is left unset.
            Pointer<MultiplyUIntModOperand> inv_root_powers_;

            ModArithLazy mod_arith_lazy_;

            NTTHandler ntt_handler_;
        };

        /**
        Allocate and construct an array of NTTTables each with different a modulus.

        @throws std::invalid_argument if modulus is empty, modulus does not support NTT, coeff_count_power is invalid,
        or pool is uninitialized.
        */
        void CreateNTTTables(
            int coeff_count_power, const std::vector<Modulus> &modulus, Pointer<NTTTables> &tables,
            MemoryPoolHandle pool);

        void ntt_negacyclic_harvey_lazy(CoeffIter operand, const NTTTables &tables);

        inline void ntt_negacyclic_harvey_lazy(
            RNSIter operand, std::size_t coeff_modulus_size, ConstNTTTablesIter tables)
        {
#ifdef SEAL_DEBUG
            if (!operand)
            {
                throw std::invalid_argument("operand");
            }
            if (!tables)
            {
                throw std::invalid_argument("tables");
            }
#endif
            SEAL_ITERATE(iter(operand, tables), coeff_modulus_size, [&](auto I) {
                ntt_negacyclic_harvey_lazy(get<0>(I), get<1>(I));
            });
        }

        inline void ntt_negacyclic_harvey_lazy(PolyIter operand, std::size_t size, ConstNTTTablesIter tables)
        {
#ifdef SEAL_DEBUG
            if (!operand)
            {
                throw std::invalid_argument("operand");
            }
            if (!tables)
            {
                throw std::invalid_argument("tables");
            }
#endif
            SEAL_ITERATE(
                operand, size, [&](auto I) { ntt_negacyclic_harvey_lazy(I, operand.coeff_modulus_size(), tables); });
        }

        void ntt_negacyclic_harvey(CoeffIter operand, const NTTTables &tables);

        inline void ntt_negacyclic_harvey(RNSIter operand, std::size_t coeff_modulus_size, ConstNTTTablesIter tables)
        {
#ifdef SEAL_DEBUG
            if (!operand)
            {
                throw std::invalid_argument("operand");
            }
            if (!tables)
            {
                throw std::invalid_argument("tables");
            }
#endif
            SEAL_ITERATE(iter(operand, tables), coeff_modulus_size, [&](auto I) {
                ntt_negacyclic_harvey(get<0>(I), get<1>(I));
            });
        }

        inline void ntt_negacyclic_harvey(PolyIter operand, std::size_t size, ConstNTTTablesIter tables)
        {
#ifdef SEAL_DEBUG
            if (!operand)
            {
                throw std::invalid_argument("operand");
            }
            if (!tables)
            {
                throw std::invalid_argument("tables");
            }
#endif
            SEAL_ITERATE(
                operand, size, [&](auto I) { ntt_negacyclic_harvey(I, operand.coeff_modulus_size(), tables); });
        }

        void inverse_ntt_negacyclic_harvey_lazy(CoeffIter operand, const NTTTables &tables);

        inline void inverse_ntt_negacyclic_harvey_lazy(
            RNSIter operand, std::size_t coeff_modulus_size, ConstNTTTablesIter tables)
        {
#ifdef SEAL_DEBUG
            if (!operand)
            {
                throw std::invalid_argument("operand");
            }
            if (!tables)
            {
                throw std::invalid_argument("tables");
            }
#endif
            SEAL_ITERATE(iter(operand, tables), coeff_modulus_size, [&](auto I) {
                inverse_ntt_negacyclic_harvey_lazy(get<0>(I), get<1>(I));
            });
        }

        inline void inverse_ntt_negacyclic_harvey_lazy(PolyIter operand, std::size_t size, ConstNTTTablesIter tables)
        {
#ifdef SEAL_DEBUG
            if (!operand)
            {
                throw std::invalid_argument("operand");
            }
            if (!tables)
            {
                throw std::invalid_argument("tables");
            }
#endif
            SEAL_ITERATE(operand, size, [&](auto I) {
                inverse_ntt_negacyclic_harvey_lazy(I, operand.coeff_modulus_size(), tables);
            });
        }

        void inverse_ntt_negacyclic_harvey(CoeffIter operand, const NTTTables &tables);

        inline void inverse_ntt_negacyclic_harvey(
            RNSIter operand, std::size_t coeff_modulus_size, ConstNTTTablesIter tables)
        {
#ifdef SEAL_DEBUG
            if (!operand)
            {
                throw std::invalid_argument("operand");
            }
            if (!tables)
            {
                throw std::invalid_argument("tables");
            }
#endif
            SEAL_ITERATE(iter(operand, tables), coeff_modulus_size, [&](auto I) {
                inverse_ntt_negacyclic_harvey(get<0>(I), get<1>(I));
            });
        }

        inline void inverse_ntt_negacyclic_harvey(PolyIter operand, std::size_t size, ConstNTTTablesIter tables)
        {
#ifdef SEAL_DEBUG
            if (!operand)
            {
                throw std::invalid_argument("operand");
            }
            if (!tables)
            {
                throw std::invalid_argument("tables");
            }
#endif
            SEAL_ITERATE(
                operand, size, [&](auto I) { inverse_ntt_negacyclic_harvey(I, operand.coeff_modulus_size(), tables); });
        }

        void ntt_negacyclic_harvey_new(CoeffIter operand, const NTTTables &tables);
        void inverse_ntt_negacyclic_harvey_new(CoeffIter operand, const NTTTables &tables);
    } // namespace util
} // namespace seal
