#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <iterator>
#include <ranges>

#include <cmath>
#include <vector>
#include <version>

#ifdef NOFORMAT
#define HAVEFORMAT 0
#else
#ifdef __cpp_lib_format
#define HAVEFORMAT 1
#endif
#endif

#ifdef NOZIP
#define HAVEZIP 0
#else
#ifdef __cpp_lib_ranges_zip
#define HAVEZIP 1
#endif
#endif

using std::vector, std::pair;

struct dvec : public vector<double> {
    // inherit constructors
    using vector::vector;

    // returns the euclidean norm
    double norm() const { return std::sqrt(dot(*this, *this)); }

    // returns the dot product of a and b.
    friend double dot(const dvec& a, const dvec& b)
    {
        assert_compatible(a, b);
        double acc = 0.0;
#ifdef HAVEZIP
        for (const auto&& [x, y] : std::views::zip(a, b)) {
            acc += x * y;
        }
#else
        for (auto i = 0u; i < a.size(); ++i) {
            acc += a[i] * b[i];
        }
#endif
        return acc;
    }

    // returns the angle theta between a and b.
    friend double theta(const dvec& a, const dvec& b)
    {
        assert_compatible(a, b);
        return std::acos(dot(a, b) / (a.norm() * b.norm()));
    }

    friend std::ostream& operator<<(std::ostream& os, const dvec& self)
    {
        if (self.size() == 0) {
            return os << "[]";
        }
        os << "[";
        auto it = self.cbegin();
        auto endm1 = std::prev(self.cend());

        for (; it != endm1; ++it) {
            os << *it << ", ";
        }
        return os << *it << "]";
    }

    friend bool operator==(const dvec&, const dvec&) = default;

private:
    static void assert_compatible(const dvec& a, const dvec& b)
    {
        if (a.size() != b.size()) {
            throw std::logic_error("mismatched dvec dimensions");
        }
    }
};

#if HAVEFORMAT
#include <format>
template<>
struct std::formatter<dvec> {
    constexpr auto parse(auto& ctx)
    {
        auto it = ctx.begin();
        if (it == ctx.end())
            return it;
        if (*it != '}')
            throw std::format_error("invalid format args");
        return it;
    }

    auto format(const dvec& v, auto& ctx) const
    {
        // gcc doesnt support formatting ranges yet :(
        std::ostringstream buf;
        buf << v;
        return std::ranges::copy(std::move(buf).str(), ctx.out()).out;
    }
};
#endif

vector<dvec> ingest_dvecs(std::istream& input)
{
    vector<dvec> output;

    for (std::string line; std::getline(input, line);) {
        std::istringstream iss{std::move(line)};
        std::istream_iterator<double> iter{iss}, end;
        dvec v;
        v.insert(v.cbegin(), iter, end);
        if (!output.empty() && output[0].size() != v.size()) {
            throw std::runtime_error("mismatched input vector dimensions");
        }
        output.push_back(std::move(v));
    }
    return output;
}

// returns all unique (i.e. [a,b]==[b,a]) pairs of vectors in `vecs`, excluding
// pairs of the same vector.
vector<pair<dvec, dvec>> pairwise_elts(const vector<dvec>& vecs)
{
    vector<pair<dvec, dvec>> pairs;

    auto it = vecs.cbegin();
    auto end = vecs.cend();
    for (; it != end; ++it) {
        for (auto it2 = it + 1; it2 != end; ++it2) {
            pairs.push_back({*it, *it2});
        }
    }
    return pairs;
}

// return the pairs of dvec2s ordered by theta in ascending order
vector<pair<dvec, dvec>> theta_sort(const vector<dvec>& vecs)
{
    auto pairs = pairwise_elts(vecs);
    std::sort(pairs.begin(), pairs.end(), [](auto x, auto y) {
        return theta(x.first, x.second) < theta(y.first, y.second);
    });
    return pairs;
}

#ifndef TESTING
constexpr std::string_view DEFAULT_FNAME = "test.txt";

int main(int argc, char** argv)
{
    std::string_view fname = DEFAULT_FNAME;
    if (argc == 2) {
        fname = argv[1];
    }

    std::ifstream ifile{fname.data()};
    if (!ifile.is_open()) {
        throw std::runtime_error("no input file");
    }

    auto vecs = ingest_dvecs(ifile);
    auto vecpairs = theta_sort(vecs);

    for (auto& [x, y] : vecpairs) {
#if HAVEFORMAT
        std::format_to(std::ostreambuf_iterator{std::cout},
                       "𝜃({}, {}) = {:f}\n", x, y, theta(x, y));
#else
        std::cout << "𝜃(" << x << ", " << y << ") = " << std::fixed
                  << theta(x, y) << std::endl
                  << std::defaultfloat;
#endif
    }
    return 0;
}
#else
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

TEST_CASE("ingest_dvecs")
{
    std::istringstream input{"1 2 3\n4 5 6\n7 8 9\n10 11 12\n13 14 15"};
    vector<dvec> expect = {
        {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}, {13, 14, 15}};

    auto result = ingest_dvecs(input);

    CHECK(result == expect);
}

TEST_CASE("theta")
{ // expected results calculated in Mathematica
    vector<dvec> input = {
        {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}, {13, 14, 15}};
    vector<double> expect = {0.225726,  0.285887,  0.313506, 0.329341,
                             0.0601607, 0.0877795, 0.103615, 0.0276188,
                             0.0434547, 0.0158359};

    auto pairs = pairwise_elts(input);

    for (int i{0}; auto p : pairs) {
        CHECK(theta(p.first, p.second) == doctest::Approx(expect[i++]));
    }
}

TEST_CASE("theta_sort")
{
    vector<dvec> input = {
        {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}, {13, 14, 15}};
    auto result = theta_sort(input);

    // ensure ascending order.
    for (double last{0.0}; auto p : result) {
        auto curr = theta(p.first, p.second);
        CHECK(last <= curr);
        last = curr;
    }
}

TEST_CASE("pairwise_elts")
{
    // just ensuring it returns the correct number of elts and that none are
    // identical.
    vector<dvec> input = {
        {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}, {13, 14, 15}};
    auto result = pairwise_elts(input);
    CHECK(result.size() == 10); // Binomial(5,2) == 10
    for (auto [x, y] : result) {
        CHECK(x != y);
    }
}

#endif
