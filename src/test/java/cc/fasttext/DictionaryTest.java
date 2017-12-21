package cc.fasttext;

import cc.fasttext.io.WordReader;
import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.stream.Collectors;

import static org.junit.Assert.assertEquals;

public class DictionaryTest {
    private static final Logger LOGGER = LoggerFactory.getLogger(DictionaryTest.class);

    private Dictionary dictionary = new Dictionary(new Args.Builder().build(), StandardCharsets.UTF_8);

    @Test
    public void testHash() {
        assertEquals(dictionary.hash(","), 688690635L);
        assertEquals(dictionary.hash("is"), 1312329493L);
        assertEquals(dictionary.hash("</s>"), 3617362777L);
    }

    @Test
    public void testFind() {
        assertEquals(dictionary.find(","), 28690635L);
        assertEquals(dictionary.find("is"), 22329493L);
        assertEquals(dictionary.find("</s>"), 17362777L);
    }

    @Test
    public void testAdd() {
        dictionary.add(",");
        dictionary.add("is");
        dictionary.add("is");
        String w = "";
        dictionary.add(w);
        dictionary.add(w);
        dictionary.add(w);
        Map<Long, Integer> word2int = dictionary.getWord2int();
        assertEquals(3, dictionary.getWords().get(word2int.get(dictionary.find(w))).count());
        assertEquals(2, dictionary.getWords().get(word2int.get(dictionary.find("is"))).count());
        assertEquals(1, dictionary.getWords().get(word2int.get(dictionary.find(","))).count());
    }

    @Test
    public void testReadWords1() throws Exception {
        Path data = Paths.get(DictionaryTest.class.getResource("/text-data.txt").toURI());
        Path words = Paths.get(DictionaryTest.class.getResource("/text-data.words").toURI());
        LOGGER.info("Data file {}", data);
        LOGGER.info("Words file {}", words);
        testReadWords(data, words);
    }

    public static void testReadWords(Path dataFile, Path wordsFile) throws Exception {
        List<String> expected = Files.lines(wordsFile).collect(Collectors.toList());
        try (WordReader r = Dictionary.createWordReader(Files.newInputStream(dataFile), StandardCharsets.UTF_8, 8 * 1024)) {
            testReadWords(expected, r::nextWord);
        }
    }

    @Test
    public void testReadWords2() throws Exception {
        Path data = Paths.get(DictionaryTest.class.getResource("/dbpedia.cut.test").toURI());
        List<String> expected = new ArrayList<>();
        try (Reader r = Files.newBufferedReader(data)) {
            String token;
            while ((token = readWord(r)) != null) {
                expected.add(token);
            }
        }
        LOGGER.debug("Expected num of words: {}", expected.size());
        List<String> actual = Dictionary.createSeekableWordReader(Files.newInputStream(data), StandardCharsets.UTF_8, 1024)
                .words().collect(Collectors.toList());
        Assert.assertEquals(expected, actual);
    }

    public static void testReadWords(List<String> expected, Callable<String> nextWord) throws Exception {
        LOGGER.debug("Expected num of words: {}", expected.size());
        for (int i = 0; ; i++) {
            String actual = nextWord.call();
            if (actual == null) {
                Assert.assertEquals("Unexpected end of stream", expected.size(), i);
                return;
            }
            String exp = expected.get(i);
            Assert.assertEquals("Wrong word #" + i, exp, actual);
        }
    }

    /**
     * Reads next word from the stream.
     * Original code (from dictionary.cc):
     * <pre>{@code bool Dictionary::readWord(std::istream& in, std::string& word) const {
     *  char c;
     *  std::streambuf& sb = *in.rdbuf();
     *  word.clear();
     *  while ((c = sb.sbumpc()) != EOF) {
     *      if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '\v' || c == '\f' || c == '\0') {
     *          if (word.empty()) {
     *              if (c == '\n') {
     *                  word += EOS;
     *                  return true;
     *              }
     *              continue;
     *          } else {
     *              if (c == '\n')
     *                  sb.sungetc();
     *              return true;
     *          }
     *      }
     *      word.push_back(c);
     *  }
     *  in.get();
     *  return !word.empty();
     * }
     * }</pre>
     *
     * @param reader, {@link java.io.Reader} markSupported = true.
     * @return String or null if the end of stream
     * @throws IOException if something is wrong
     */
    public static String readWord(java.io.Reader reader) throws IOException {
        StringBuilder sb = new StringBuilder();
        int i;
        while ((i = reader.read()) != -1) {
            char c = (char) i;
            if (c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == 0x000b || c == '\f' || c == '\0') {
                if (sb.length() == 0) {
                    if (c == '\n') {
                        return Dictionary.EOS;
                    }
                    continue;
                } else {
                    if (c == '\n') {
                        reader.reset();
                    }
                    return sb.toString();
                }
            }
            reader.mark(1);
            sb.append(c);
        }
        return sb.length() == 0 ? null : sb.toString();
    }

}
