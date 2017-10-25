#include <iostream>
#include <fstream>

#include <string>
#include <vector>
#include <stdlib.h>

class Utils {
public:
  Utils();
  int64_t size(std::ifstream&);
  void seek(std::ifstream&, int64_t);
  bool readWord(std::istream&, std::string&) const;
};

Utils::Utils() {
}

int64_t Utils::size(std::ifstream& ifs) {
    ifs.seekg(std::streamoff(0), std::ios::end);
    return ifs.tellg();
}

void Utils::seek(std::ifstream& ifs, int64_t pos) {
    ifs.clear();
    ifs.seekg(std::streampos(pos));
}

bool Utils::readWord(std::istream& in, std::string& word) const
{
  char c;
  std::streambuf& sb = *in.rdbuf();
  word.clear();
  while ((c = sb.sbumpc()) != EOF) {
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' ||
        c == '\f' || c == '\0') {
      if (word.empty()) {
        if (c == '\n') {
          word += "</s>";
          return true;
        }
        continue;
      } else {
        if (c == '\n')
          sb.sungetc();
        return true;
      }
    }
    word.push_back(c);
  }
  // trigger eofbit
  in.get();
  return !word.empty();
}

int main(int argc, char** argv)
{
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() < 2) {
        std::cerr << "Wrong input" << std::endl;
        exit(-2);
    }
    std::string infile = args[1];
    std::cout << "infile:\t" << infile << "\n";

    std::ifstream ifs(infile);

    if (ifs.is_open()) {
        Utils utils;
        std::string word;
        int64_t count = 1;
        std::vector<std::string> words;
        while (utils.readWord(ifs, word)) {
            count++;
            words.push_back(word);
        }
        std::cout << "count:\t" <<  count << "\n";

        std::cout << "[" << std::endl;
        for (auto i = words.begin(); i != words.end(); ++i)
            std::cout << *i << ", ";
        std::cout << "]" << std::endl;
        ifs.close();
    } else {
        std::cerr << "Test file cannot be opened!" << std::endl;
    }
    return 0;
}