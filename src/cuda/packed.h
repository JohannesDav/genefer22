#pragma once
#include "packed_device.h"
#include "../transformGPU.h"
#include <memory>
// Pack a power-of-two number of base-b digits per coefficient for squaring. General operations
// and checkpoint storage retain the selected upstream two- or three-prime transform.
template<size_t Fields = 3, bool Wide = false>
class packed_transform : public transformGPUs<Fields, Wide> {
    using baseline = transformGPUs<Fields, Wide>;
    mutable std::unique_ptr<packed_device> packed_;
    mutable bool packed_current_ = false, baseline_current_ = true;
    bool proof_generation_ = false;
    mutable std::vector<int32_t> digits_;
    void materialize() const {
        if (packed_ && !baseline_current_) {
            packed_->getDigits(digits_.data());
            const_cast<packed_transform *>(this)->baseline::setZi(digits_.data());
            baseline_current_ = true;
        }
    }
    void prepare() {
        if (!packed_current_) {
            baseline::getZi(digits_.data());
            packed_->setDigits(digits_.data());
            packed_current_ = true;
        }
    }

  protected:
    void getZi(int32_t *out) const override {
        materialize();
        baseline::getZi(out);
    }
    void setZi(const int32_t *in) override {
        if (packed_)
            packed_->synchronize();
        baseline::setZi(in);
        packed_current_ = false;
        baseline_current_ = true;
    }

  public:
    packed_transform(uint32_t b, uint32_t n, bool boinc, size_t device, size_t registers,
                     cl_platform_id platform_id, cl_device_id device_id, bool verbose,
                     const std::string &embedded_kernel = {}, int shared_memory_preference = 50)
        : baseline(b, n, boinc, device, registers, platform_id, device_id, verbose),
          digits_(size_t(1) << n) {
        const char *path = embedded_kernel.empty() ? nullptr : embedded_kernel.c_str();
#if !defined(GENEFER_AUTOMATIC_CUDA)
        // Developer probes can load generated kernels; the shipped automatic
        // selector uses only sources admitted and generated inside the binary.
        if (!path) path = std::getenv("GENEFER_PACKED_KERNEL");
#endif
        if (path) {
            if (n < 16 || n > 22)
                throw std::runtime_error("Packed CUDA arithmetic requires a supported GFN exponent and device");
            if (boinc && platform_id && device_id)
                packed_ = std::make_unique<packed_device>(b, device_id, path, verbose, size_t(1) << n,
                                                         !embedded_kernel.empty(), shared_memory_preference);
            else {
                platform parent;
                packed_ = std::make_unique<packed_device>(b, nvidia_device(parent, device),
                                                          path, verbose, size_t(1) << n,
                                                          !embedded_kernel.empty(), shared_memory_preference);
            }
        }
    }
    void synchronizeForCalibration() const override {
        if (packed_) packed_->synchronize();
        baseline::synchronizeForCalibration();
    }
    void setProofGeneration(bool enabled) override {
        if (enabled) {
            materialize();
            packed_current_ = false;
        }
        proof_generation_ = enabled;
    }
    void squareDup(bool dup) override {
        if (!packed_ || proof_generation_) {
            baseline::squareDup(dup);
            packed_current_ = false;
            baseline_current_ = true;
            return;
        }
        prepare();
        packed_->square(dup);
        baseline_current_ = false;
    }
    void squareMul(int32_t a) override {
        if (a < 1)
            throw std::runtime_error("Scalar must be positive");
        if (a <= 2) {
            squareDup(a == 2);
            return;
        }
        materialize();
        baseline::squareMul(a);
        packed_current_ = false;
        baseline_current_ = true;
    }
    void set(uint32_t a) override {
        if (packed_)
            packed_->synchronize();
        baseline::set(a);
        packed_current_ = false;
        baseline_current_ = true;
    }
    void initMultiplicand(size_t src) override {
        if (src == 0)
            materialize();
        baseline::initMultiplicand(src);
    }
    void mul() override {
        materialize();
        baseline::mul();
        packed_current_ = false;
        baseline_current_ = true;
    }
    void copy(size_t dst, size_t src) const override {
        if (src == 0)
            materialize();
        if (dst == 0 && packed_)
            packed_->synchronize();
        baseline::copy(dst, src);
        if (dst == 0) {
            packed_current_ = false;
            baseline_current_ = true;
        }
    }
    bool readContext(file &f, size_t count) override {
        if (packed_)
            packed_->synchronize();
        bool ok = baseline::readContext(f, count);
        packed_current_ = false;
        baseline_current_ = true;
        return ok;
    }
    void saveContext(file &f, size_t count) const override {
        materialize();
        baseline::saveContext(f, count);
    }
    size_t getMemSize() const override {
        return baseline::getMemSize() + (packed_ ? packed_->memory() : 0) + digits_.size() * 4;
    }
};
