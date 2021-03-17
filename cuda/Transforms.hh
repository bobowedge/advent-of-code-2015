#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

const size_t TSIZE = 43;

std::map<std::string, std::vector<std::string>> TRANSFORMS_MAP = {
    std::make_pair(std::string("Al"), std::vector<std::string>{std::string("ThF"), std::string("ThRnFAr")}),
    std::make_pair(std::string("B"), std::vector<std::string>{std::string("BCa"), std::string("TiB"), std::string("TiRnFAr")}),
    std::make_pair(std::string("Ca"), std::vector<std::string>{std::string("CaCa"), std::string("PB"), std::string("PRnFAr"), std::string("SiRnFYFAr"), std::string("SiRnMgAr"), std::string("SiTh")}),
    std::make_pair(std::string("F"), std::vector<std::string>{std::string("CaF"), std::string("PMg"), std::string("SiAl")}),
    std::make_pair(std::string("H"), std::vector<std::string>{std::string("CRnAlAr"), std::string("CRnFYFYFAr"), std::string("CRnFYMgAr"), std::string("CRnMgYFAr"), std::string("HCa"), std::string("NRnFYFAr"), std::string("NRnMgAr"), std::string("NTh"), std::string("OB"), std::string("ORnFAr")}),
    std::make_pair(std::string("Mg"),std::vector<std::string> {std::string("BF"), std::string("TiMg")}),
    std::make_pair(std::string("N"), std::vector<std::string>{std::string("CRnFAr"), std::string("HSi")}),
    std::make_pair(std::string("O"), std::vector<std::string>{std::string("CRnFYFAr"), std::string("CRnMgAr"), std::string("HP"), std::string("NRnFAr"), std::string("OTi")}),
    std::make_pair(std::string("P"), std::vector<std::string>{std::string("CaP"), std::string("PTi"), std::string("SiRnFAr")}),
    std::make_pair(std::string("Si"), std::vector<std::string>{std::string("CaSi")}),
    std::make_pair(std::string("Th"), std::vector<std::string>{std::string("ThCa")}),
    std::make_pair(std::string("Ti"), std::vector<std::string>{std::string("BP"), std::string("TiTi")}),
    std::make_pair(std::string("E"), std::vector<std::string>{std::string("HF"), std::string("NAl"), std::string("OMg")})
};

const std::string MOLECULE_STRING = "CRnCaSiRnBSiRnFArTiBPTiTiBFArPBCaSiThSiRnTiBPBPMgArCaSiRnTiMgArCaSiThCaSiRnFArRnSiRnFArTiTiBFArCaCaSiRnSiThCaCaSiRnMgArFYSiRnFYCaFArSiThCaSiThPBPTiMgArCaPRnSiAlArPBCaCaSiRnFYSiThCaRnFArArCaCaSiRnPBSiRnFArMgYCaCaCaCaSiThCaCaSiAlArCaCaSiRnPBSiAlArBCaCaCaCaSiThCaPBSiThPBPBCaSiRnFYFArSiThCaSiRnFArBCaCaSiRnFYFArSiThCaPBSiThCaSiRnPMgArRnFArPTiBCaPRnFArCaCaCaCaSiRnCaCaSiRnFYFArFArBCaSiThFArThSiThSiRnTiRnPMgArFArCaSiThCaPBCaSiRnBFArCaCaPRnCaCaPMgArSiRnFYFArCaSiThRnPBPMgAr";

const std::vector<std::string> ELEMENTS = {std::string("Al"), std::string("Ar"), std::string("B"), std::string("C"), std::string("Ca"), std::string("F"), std::string("H"), 
                                           std::string("Mg"), std::string("N"), std::string("O"), std::string("P"), std::string("Rn"), std::string("Si"), std::string("Th"), 
                                           std::string("Ti"), std::string("Y"), std::string("E")};

__device__ __managed__ size_t TSIZES[TSIZE] = {3, 5, 3, 3, 5, 3, 3, 5, 7, 5, 3, 3, 3, 3, 3, 3, 3, 5, 9, 7, 7, 3, 7, 5, 3, 3, 5, 3, 3, 5, 3, 7, 5, 3, 5, 3, 3, 3, 5, 3, 3, 3, 3};
__device__ __managed__ int64_t TRANSFORMS[TSIZE][9] = {
    {0, 13, 5, -1, -1, -1, -1, -1, -1},
    {0, 13, 11, 5, 1, -1, -1, -1, -1},
    {2, 2, 4, -1, -1, -1, -1, -1, -1},
    {2, 14, 2, -1, -1, -1, -1, -1, -1},
    {2, 14, 11, 5, 1, -1, -1, -1, -1},
    {4, 4, 4, -1, -1, -1, -1, -1, -1},
    {4, 10, 2, -1, -1, -1, -1, -1, -1},
    {4, 10, 11, 5, 1, -1, -1, -1, -1},
    {4, 12, 11, 5, 15, 5, 1, -1, -1},
    {4, 12, 11, 7, 1, -1, -1, -1, -1},
    {4, 12, 13, -1, -1, -1, -1, -1, -1},
    {16, 6, 5, -1, -1, -1, -1, -1, -1},
    {16, 8, 0, -1, -1, -1, -1, -1, -1},
    {16, 9, 7, -1, -1, -1, -1, -1, -1},
    {5, 4, 5, -1, -1, -1, -1, -1, -1},
    {5, 10, 7, -1, -1, -1, -1, -1, -1},
    {5, 12, 0, -1, -1, -1, -1, -1, -1},
    {6, 3, 11, 0, 1, -1, -1, -1, -1},
    {6, 3, 11, 5, 15, 5, 15, 5, 1},
    {6, 3, 11, 5, 15, 7, 1, -1, -1},
    {6, 3, 11, 7, 15, 5, 1, -1, -1},
    {6, 6, 4, -1, -1, -1, -1, -1, -1},
    {6, 8, 11, 5, 15, 5, 1, -1, -1},
    {6, 8, 11, 7, 1, -1, -1, -1, -1},
    {6, 8, 13, -1, -1, -1, -1, -1, -1},
    {6, 9, 2, -1, -1, -1, -1, -1, -1},
    {6, 9, 11, 5, 1, -1, -1, -1, -1},
    {7, 2, 5, -1, -1, -1, -1, -1, -1},
    {7, 14, 7, -1, -1, -1, -1, -1, -1},
    {8, 3, 11, 5, 1, -1, -1, -1, -1},
    {8, 6, 12, -1, -1, -1, -1, -1, -1},
    {9, 3, 11, 5, 15, 5, 1, -1, -1},
    {9, 3, 11, 7, 1, -1, -1, -1, -1},
    {9, 6, 10, -1, -1, -1, -1, -1, -1},
    {9, 8, 11, 5, 1, -1, -1, -1, -1},
    {9, 9, 14, -1, -1, -1, -1, -1, -1},
    {10, 4, 10, -1, -1, -1, -1, -1, -1},
    {10, 10, 14, -1, -1, -1, -1, -1, -1},
    {10, 12, 11, 5, 1, -1, -1, -1, -1},
    {12, 4, 12, -1, -1, -1, -1, -1, -1},
    {13, 13, 4, -1, -1, -1, -1, -1, -1},
    {14, 2, 10, -1, -1, -1, -1, -1, -1},
    {14, 14, 14, -1, -1, -1, -1, -1, -1}
};

std::vector<int64_t> to_int64(const std::string& x)
{
    std::vector<int64_t> output;
    size_t idx = 0;
    while (idx < x.size())
    {
        std::string elem = x.substr(idx, 1);
        if (idx + 1 < x.size() && x[idx + 1] >= 'a' && x[idx + 1] <= 'z')
        {
            elem = x.substr(idx, 2);
            ++idx;
        }
        ++idx;
        auto it = std::find(ELEMENTS.begin(), ELEMENTS.end(), elem);
        output.push_back(it - ELEMENTS.begin());
    }
    return output;
}

std::vector<std::vector<int64_t>> convert_transforms(const std::map<std::string, std::vector<std::string>>& t)
{
    std::vector<std::vector<int64_t>> output;
    for (const auto& p : t)
    {
        auto key = to_int64(p.first);
        for(const auto& values : p.second)
        {            
            auto row = key;
            auto converted = to_int64(values);
            row.insert(row.end(), converted.begin(), converted.end());
            output.push_back(row);
        }
    }
    return output;
}