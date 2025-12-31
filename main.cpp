#include <iostream>
#include <fcntl.h>
#include <io.h>

#include "CommonUtils.h" 

#include "KBRulebase.h" 
#include "BarcodeRulebase.h"

int main()
{
    _setmode(_fileno(stdout), _O_U16TEXT);

    std::wstring kb_path = L"E:/Dev/Hankook/ky_contour_extractor/test1";
    std::wstring barcode_path = L"E:/Dev/Hankook/barcode_test_images"; 

    std::wcout << L"--- Inspection Program Started ---" << std::endl;
    // Run_KB_Rulebase_Algorithm(kb_path);

    std::wcout << L">>> Running Barcode Inspection..." << std::endl;
    Run_Barcode_Rulebase_Algorithm(barcode_path);

    std::wcout << L"--- All Finished ---" << std::endl;
    return 0;
}