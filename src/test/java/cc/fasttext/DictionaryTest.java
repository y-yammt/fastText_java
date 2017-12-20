package cc.fasttext;

import org.junit.Assert;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
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
    public void testReadWords() throws Exception {
        Path data = Paths.get(DictionaryTest.class.getResource("/text-data.txt").toURI());
        Path words = Paths.get(DictionaryTest.class.getResource("/text-data.words").toURI());
        LOGGER.info("Data file {}", data);
        LOGGER.info("Words file {}", words);
        testReadWords(data, words);
    }

    public static void testReadWords(Path dataFile, Path wordsFile) throws Exception {
        List<String> expected = Files.lines(wordsFile).collect(Collectors.toList());
        try (Reader r = Files.newBufferedReader(dataFile)) {
            testReadWords(expected, () -> Dictionary.readWord(r));
        }
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


}
