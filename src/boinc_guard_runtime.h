#pragma once
// Linux native BOINC coordinator; GPU arithmetic/proof engine is unchanged.

#include <cerrno>
#include <cstdint>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <spawn.h>
#include <string>
#include <stdexcept>
#include <system_error>
#include <vector>
#include <chrono>
#include <thread>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#if !defined(BOINC) || !defined(GENEFER_CUDA) || !defined(GPU)
#error The guarded coordinator requires the real BOINC API and CUDA GPU build.
#endif
#if defined(GENEFER_GUARD_3090) == defined(GENEFER_GUARD_4060)
#error Select exactly one GFN-19 GPU profile.
#endif

extern char **environ;

namespace boinc_guard
{
namespace fs = std::filesystem;
enum class outcome
{
    success,
    interrupted,
    failed
};
static volatile std::sig_atomic_t signal_quit = 0;
inline void request_quit()
{
    signal_quit = 1;
}

inline bool cancelled()
{
    BOINC_STATUS status{};
    boinc_get_status(&status);
    return signal_quit || status.quit_request || status.abort_request || status.no_heartbeat;
}

inline void require(bool condition, const std::string &message)
{
    if (!condition)
        throw std::runtime_error("BOINC guard: " + message);
}

inline fs::path physical_target(const std::string &logical)
{
    char buffer[4096]{};
    require(boinc_resolve_filename(logical.c_str(), buffer, sizeof(buffer)) == 0,
            "cannot resolve output " + logical);
    fs::path path = fs::absolute(buffer).lexically_normal();
    // Preserve slot symlinks, including links whose destination is not created
    // yet. weakly_canonical() alone is insufficient for a dangling last link.
    for (int links = 0; links < 40; ++links)
    {
        std::error_code error;
        auto status = fs::symlink_status(path, error);
        require(!error || error == std::errc::no_such_file_or_directory,
                "cannot inspect output target " + path.string());
        if (!fs::is_symlink(status))
        {
            require(fs::is_directory(path.parent_path()), "output directory missing");
            return path;
        }
        auto target = fs::read_symlink(path);
        path = (target.is_absolute() ? target : path.parent_path() / target).lexically_normal();
    }
    throw std::runtime_error("BOINC guard: output symlink loop");
}

inline void sync_file(const fs::path &path)
{
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    require(fd >= 0, "cannot open output for sync");
    int result = ::fsync(fd);
    int saved = errno;
    ::close(fd);
    require(result == 0, "cannot sync output: " + std::string(std::strerror(saved)));
}

inline void sync_directory(const fs::path &path)
{
    int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    require(fd >= 0, "cannot open output directory for sync");
    int result = ::fsync(fd);
    int saved = errno;
    ::close(fd);
    require(result == 0, "cannot sync output directory: " + std::string(std::strerror(saved)));
}

inline void atomic_text(const fs::path &destination, const std::string &text)
{
    auto temporary = destination.string() + ".guard-new-" + std::to_string(::getpid());
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        require(bool(output), "cannot create temporary result");
        output.write(text.data(), std::streamsize(text.size()));
        output.close();
        require(bool(output), "cannot finish temporary result");
    }
    sync_file(temporary);
    fs::rename(temporary, destination);
    sync_directory(destination.parent_path());
}

inline void atomic_copy(const fs::path &source, const fs::path &destination)
{
    auto temporary = destination.string() + ".guard-new-" + std::to_string(::getpid());
    fs::copy_file(source, temporary, fs::copy_options::overwrite_existing);
    sync_file(temporary);
    fs::rename(temporary, destination);
    sync_directory(destination.parent_path());
}

class coordinator
{
    fs::path bundle_, prefix_;
    std::string public_prefix_;
    uint32_t base_;
    size_t device_;
    bool check_mode_;
    bool check_certificate_ = false, check_tail_ = false;

    static void install_environment(const fs::path &bundle)
    {
        std::vector<std::string> clear;
        for (char **entry = environ; entry && *entry; ++entry)
        {
            std::string value(*entry);
            auto equal = value.find('=');
            auto name = value.substr(0, equal);
            if (name.rfind("GENEFER_", 0) == 0 || name.rfind("CUDA_", 0) == 0)
                clear.push_back(name);
        }
        for (const auto &name : clear)
            ::unsetenv(name.c_str());
        auto set = [](const char *name, const std::string &value) {
            require(::setenv(name, value.c_str(), 1) == 0, "cannot set GPU runtime policy");
        };
#if defined(GENEFER_THREE_PRIME)
        set("GENEFER_ARITHMETIC", "three");
        set("GENEFER_KERNEL", (bundle / "kernels/gfn19-three.cl").string());
#elif defined(GENEFER_EXACT_TWO)
        set("GENEFER_ARITHMETIC", "two");
        set("GENEFER_KERNEL", (bundle / "kernels/gfn19-two.cl").string());
#else
        set("GENEFER_ARITHMETIC", "upstream");
        set("GENEFER_KERNEL", (bundle / "kernels/gfn19-two.cl").string());
#endif
    }

    static void signal_group(pid_t child, int signal)
    {
        if (::kill(-child, signal) != 0 && errno != ESRCH)
            pio::error("BOINC guard: could not signal verifier process group");
    }

    static void stop_child(pid_t child)
    {
        signal_group(child, SIGCONT);
        signal_group(child, SIGINT);
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        int status = 0;
        while (std::chrono::steady_clock::now() < deadline)
        {
            auto result = ::waitpid(child, &status, WNOHANG);
            if (result == child || (result < 0 && errno == ECHILD))
            {
                // Do not leave an OpenCL grandchild running if its Python
                // supervisor exited first during an interrupted verification.
                signal_group(child, SIGKILL);
                return;
            }
            if (result < 0 && errno != EINTR)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        signal_group(child, SIGKILL);
        while (::waitpid(child, &status, 0) < 0 && errno == EINTR)
        {
        }
    }

    outcome helper(const std::string &mode, bool proof_ready)
    {
        if (cancelled())
            return outcome::interrupted;
        std::vector<std::string> words{"/usr/bin/python3",
                                       "-I",
                                       "-B",
                                       (bundle_ / "boinc_guard_verify.py").string(),
                                       "--mode",
                                       mode,
                                       "--bundle",
                                       bundle_.string(),
                                       "--prefix",
                                       prefix_.string(),
                                       "--operation",
                                       check_mode_ ? "check" : "proof",
                                       "--base",
                                       std::to_string(base_),
                                       "--device",
                                       std::to_string(device_)};
        std::vector<char *> args;
        for (auto &word : words)
            args.push_back(word.data());
        args.push_back(nullptr);
        auto log = prefix_.string() + ".helper-" + mode + ".log";
        posix_spawn_file_actions_t actions;
        posix_spawnattr_t attributes;
        require(::posix_spawn_file_actions_init(&actions) == 0, "cannot initialize child I/O");
        require(::posix_spawnattr_init(&attributes) == 0, "cannot initialize child attributes");
        bool configured =
            ::posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0) == 0 &&
            ::posix_spawn_file_actions_addopen(&actions, 1, log.c_str(),
                                               O_WRONLY | O_CREAT | O_APPEND, 0600) == 0 &&
            ::posix_spawn_file_actions_adddup2(&actions, 1, 2) == 0;
#if defined(__GLIBC__) && __GLIBC_PREREQ(2, 34)
        configured = configured && ::posix_spawn_file_actions_addclosefrom_np(&actions, 3) == 0;
#else
#error The BOINC coordinator requires Linux with glibc 2.34 or later.
#endif
        sigset_t empty_mask, default_signals;
        ::sigemptyset(&empty_mask);
        ::sigemptyset(&default_signals);
        ::sigaddset(&default_signals, SIGINT);
        ::sigaddset(&default_signals, SIGTERM);
        ::sigaddset(&default_signals, SIGPIPE);
        configured =
            configured &&
            ::posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGMASK |
                                                        POSIX_SPAWN_SETSIGDEF) == 0 &&
            ::posix_spawnattr_setpgroup(&attributes, 0) == 0 &&
            ::posix_spawnattr_setsigmask(&attributes, &empty_mask) == 0 &&
            ::posix_spawnattr_setsigdefault(&attributes, &default_signals) == 0;
        pid_t child = -1;
        int error =
            configured ? ::posix_spawn(&child, args[0], &actions, &attributes, args.data(), environ)
                       : EINVAL;
        ::posix_spawn_file_actions_destroy(&actions);
        ::posix_spawnattr_destroy(&attributes);
        require(error == 0, "cannot launch private verifier: " + std::string(std::strerror(error)));
        bool stopped = false;
        auto previous_tick = std::chrono::steady_clock::now();
        auto active_wait = std::chrono::steady_clock::duration::zero();
        for (;;)
        {
            int status = 0;
            auto result = ::waitpid(child, &status, WNOHANG);
            if (result == child)
            {
                if (cancelled())
                    return outcome::interrupted;
                return WIFEXITED(status) && WEXITSTATUS(status) == 0 ? outcome::success
                                                                     : outcome::failed;
            }
            if (result < 0 && errno != EINTR)
            {
                stop_child(child);
                return outcome::failed;
            }
            BOINC_STATUS state{};
            boinc_get_status(&state);
            if (signal_quit || state.quit_request || state.abort_request || state.no_heartbeat)
            {
                stop_child(child);
                return outcome::interrupted;
            }
            if (bool(state.suspended) != stopped)
            {
                signal_group(child, state.suspended ? SIGSTOP : SIGCONT);
                stopped = bool(state.suspended);
            }
            // The completed native proof and its completed PRP context already
            // exist atomically. Do not acknowledge an in-progress prepare step.
            if (proof_ready && boinc_time_to_checkpoint())
            {
                if (!fs::is_regular_file(prefix_.string() + ".proof") ||
                    !fs::is_regular_file(prefix_.string() + ".ctx"))
                {
                    stop_child(child);
                    return outcome::failed;
                }
                boinc_checkpoint_completed();
            }
            // Observation timeout does not restart any computation. It fails
            // this private helper closed; all durable proof/context files stay.
            auto tick = std::chrono::steady_clock::now();
            if (!stopped)
                active_wait += tick - previous_tick;
            previous_tick = tick;
            if (active_wait > std::chrono::minutes(30))
            {
                stop_child(child);
                return outcome::failed;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

  public:
    coordinator(const std::vector<std::string> &arguments, uint32_t base, uint32_t n, size_t device,
                const std::string &public_prefix, bool certificate_check = false)
        : public_prefix_(public_prefix.empty() ? "g19_" + std::to_string(base) : public_prefix),
          base_(base), device_(device), check_mode_(certificate_check)
    {
        const uint32_t maximum = 2000000000u;
        require(n == 19 && device == 0 && base >= 1024 && base <= maximum && base % 2 == 0 &&
                    (base & (base - 1)) != 0,
                "request outside the single-GPU GFN-19 specialization");
        for (size_t index = 0; index < arguments.size(); ++index)
        {
            if (arguments[index] == "--check-certificate") check_certificate_ = true;
            if (arguments[index] == "--check-tail") check_tail_ = true;
            require(arguments[index] != "--two-prime", "--two-prime was removed");
            if (arguments[index] == "--guard-bundle")
            {
                require(bundle_.empty() && index + 1 < arguments.size(),
                        "invalid --guard-bundle option");
                bundle_ = fs::canonical(arguments[++index]);
            }
        }
        require(!bundle_.empty() && fs::is_directory(bundle_), "--guard-bundle assets missing");
        prefix_ = fs::current_path() /
                  (std::string(check_mode_ ? ".gfn_checked_c_19_" : ".gfn_checked_19_") +
                   std::to_string(base));
        install_environment(bundle_);
        require(::setenv("GENEFER_CHECK_CERTIFICATE", check_certificate_ ? "1" : "0", 1) == 0 &&
                ::setenv("GENEFER_CHECK_TAIL", check_tail_ ? "1" : "0", 1) == 0,
                "cannot install verification policy");
    }

    std::string private_prefix() const
    {
        return prefix_.filename().string();
    }
    outcome prepare()
    {
        const auto result = helper("prepare", false);
#if defined(GENEFER_THREE_PRIME)
        if (result == outcome::success && !check_mode_)
        {
            const auto path = prefix_.string() + ".packed.cl";
            if (fs::exists(path))
                require(::setenv("GENEFER_PACKED_KERNEL", path.c_str(), 1) == 0,
                        "cannot install packed kernel path");
        }
#endif
        return result;
    }

    outcome finish(const std::string &result_text)
    {
        require(!check_mode_, "wrong operation for optimized proof publication");
        if (cancelled())
            return outcome::interrupted;
        require(!result_text.empty(), "native result was not quarantined");
        atomic_text(prefix_.string() + ".native-result.txt", result_text);
        auto verified = helper("verify", true);
        if (verified != outcome::success)
            return verified;
        if (cancelled())
            return outcome::interrupted;
        // Publish only after the requested checks and input-identity checks pass.
        auto public_proof = physical_target(public_prefix_ + ".proof");
        auto public_result = physical_target("results.txt");
        require(public_proof != public_result, "output aliases collide");
        atomic_copy(prefix_.string() + ".proof", public_proof);
        if (cancelled())
            return outcome::interrupted;
        // Replacement, not append: safe to repeat after interruption between
        // publication and the native parent's boinc_finish(0).
        atomic_text(public_result, result_text);
        if (cancelled())
            return outcome::interrupted;
        pio::print("BOINC guard: completed proof and result published.\n");
        return outcome::success;
    }

    outcome check_certificate()
    {
        require(check_mode_, "wrong operation for conservative certificate check");
        if (cancelled())
            return outcome::interrupted;
        auto input = physical_target(public_prefix_ + ".cert");
        require(fs::is_regular_file(input), "mapped input certificate missing");
        atomic_copy(input, prefix_.string() + ".cert");
        auto checked = helper("check", false);
        if (checked != outcome::success)
            return checked;
        if (cancelled())
            return outcome::interrupted;
        std::ifstream stream(prefix_.string() + ".conservative-result.txt", std::ios::binary);
        require(bool(stream), "private conservative result missing");
        std::string text{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
        require(stream.eof() || bool(stream), "cannot read conservative result");
        require(!text.empty(), "empty conservative result");
        // The small checker restarts from its immutable input on interruption;
        // successful result/input files are durable before this acknowledgement.
        if (boinc_time_to_checkpoint())
            boinc_checkpoint_completed();
        atomic_text(physical_target("results.txt"), text);
        if (cancelled())
            return outcome::interrupted;
        pio::print("BOINC guard: conservative certificate result published.\n");
        return outcome::success;
    }
};
} // namespace boinc_guard
