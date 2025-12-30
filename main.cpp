#include <iostream>
#include <fcntl.h>
#include <io.h>
#include "KBRulebase.h"

int main() {
    // Set console mode to support Unicode (for Korean log output compatibility if needed)
    _setmode(_fileno(stdout), _O_U16TEXT);

    // Target directory path
    std::wstring kb_path = L"E:/Dev/Hankook/ky_contour_extractor/test1";

    std::wcout << L"Starting KB Rulebase Inspection..." << std::endl;

    // Call the encapsulated function
    Run_KB_Rulebase_Algorithm(kb_path);

    std::wcout << L"Program Finished." << std::endl;
    return 0;
}