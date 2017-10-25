#include <stdint.h>
#include <iostream>
#include <string>
#include <vector>

class TestClass {
public:
  TestClass();
  uint32_t test(const std::string& str) const;
};

TestClass::TestClass() {
}

uint32_t TestClass::test(const std::string& str) const
{
  uint32_t res = 0;
  for (int i = 0; i < str.size(); i++) {
    if ((str[i] & 0xC0) == 0x80) continue;
    res++;
  }
  return res;
}

int main()
{
  const TestClass tester;

  std::vector<std::string> test;
  test.push_back("");
  test.push_back("a");
  test.push_back("Test");
  test.push_back("This is some test sentence.");
  test.push_back("这是一些测试句子。");
  test.push_back("Šis ir daži pārbaudes teikumi.");
  test.push_back("Тестовое предложение");
  test.push_back("Получение положительного заключения испытательной лаборатории по результатам сертификационных испытаний ИСУ ОПК на соответствие требованиям информационной безопасности.");

  for (int i = 0; i < test.size(); ++i) {
    std::string w = test[i];
    uint32_t h = tester.test(w);
    std::cout << "" << h << "\t'" << w << "'\n";
  }
  return 0;
}