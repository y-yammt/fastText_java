package ru.avicomp.tests;

import cc.fasttext.Main;
import org.junit.Assert;
import org.junit.FixMethodOrder;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.MethodSorters;
import org.junit.runners.Parameterized;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.avicomp.TestsBase;

import java.io.*;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static ru.avicomp.TestsBase.DESTINATION_DIR;
import static ru.avicomp.TestsBase.compareVectors;

/**
 * Created by @szuev on 20.10.2017.
 */
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
@RunWith(Parameterized.class)
public class SimpleModelTest {
    private static final Logger LOGGER = LoggerFactory.getLogger(SimpleModelTest.class);

    private final Data data;

    public SimpleModelTest(Data data) {
        this.data = data;
    }

    @Parameterized.Parameters(name = "{0}")
    public static List<Data> getData() {
        return Arrays.asList(Data.values());
    }


    @Test
    public void test01Train() throws Exception {
        Main.train(data.command());

        Path bin = data.getModelBin();
        Path vec = data.getModelVec();
        Assert.assertTrue("No .bin", Files.exists(bin));
        Assert.assertTrue("No .vec", Files.exists(vec));

        // validate bin:
        long actualBinSize = Files.size(bin);
        Assert.assertEquals("Incorrect bin size: " + actualBinSize, data.binSize(), actualBinSize);

        // validate vec:
        int allowableDiffInPercents = 10;
        long actualVecSize = Files.size(vec);
        double actualDiffInPercents = 200.0 * (actualVecSize - data.vecSize()) / (actualVecSize + data.vecSize());
        LOGGER.info(String.format("Actual vec diff: %.2f%% (size: %d)", actualDiffInPercents, actualVecSize));
        Assert.assertTrue("Incorrect vec size: " + actualVecSize + ", diff: " + actualDiffInPercents, Math.abs(actualDiffInPercents) <= allowableDiffInPercents);
        List<Word> words = collect(vec);
        LOGGER.info("{}", toSet(words));
        Assert.assertEquals("Wrong size", data.vecWords(), words.size());
        Assert.assertTrue("Wrong dim inside file", toMap(words).values().stream().allMatch(floats -> floats.size() == data.vecDim()));
        try (BufferedReader r = Files.newBufferedReader(vec)) {
            Assert.assertEquals("Wrong first line", data.vecWords() + " " + data.vecDim(), r.lines().findFirst().orElseThrow(AssertionError::new));
        }
    }

    @Test
    public void test02Predict() throws Exception {
        String cmd = "predict %s %s";

        Path in = Paths.get(SimpleModelTest.class.getResource("/dbpedia.cut.test").toURI());

        ByteArrayOutputStream array = new ByteArrayOutputStream();
        PrintStream newOut = new PrintStream(array);
        PrintStream out = System.out;
        try {
            System.setOut(newOut);
            Main.predict(TestsBase.cmd(cmd, data.getModelBin(), in));
        } catch (IllegalArgumentException e) {
            if (!Data.SUPERVISED_THREAD4_DIM10_LR01_NGRAMS2_BUCKET1E7_EPOCH5.equals(data)) {
                LOGGER.debug("Expected exception: '{}'", e.getMessage());
                return;
            }
        } finally {
            System.setOut(out);
        }
        List<String> actual = Arrays.stream(array.toString(StandardCharsets.UTF_8.name()).split("\r*\n")).collect(Collectors.toList());
        List<String> expected = Files.lines(Paths.get(SimpleModelTest.class.getResource("/simple.predict.result").toURI())).collect(Collectors.toList());
        actual.stream()
                .limit(10)
                .forEach(s -> LOGGER.debug("{}", s));
        LOGGER.info("Size: {}", actual.size());
        Assert.assertEquals("Wrong count of lines in out", 161, actual.size());
        // the test set is small, the result depending on current model:
        TestsBase.compareLists(expected, actual, 50);
    }

    @Test
    public void test04PrintSentenceVectors() throws Exception {
        String str = runPrintVectors(data.getModelBin(), data.getSentenceVectorsTestData(), true);
        LOGGER.info("Output: {}", str);
        List<Double> res = Arrays.stream(str.split("\\s+"))
                .mapToDouble(Double::parseDouble)
                .boxed()
                .collect(Collectors.toList());
        Assert.assertEquals(data.vecDim(), res.size());
        compareVectors(data.sentenceVectors(), res, data.vecDelta());
    }

    @Test
    public void test03PrintWordVectors() throws Exception {
        String str = runPrintVectors(data.getModelBin(), data.getWordVectorsTestData(), false);
        LOGGER.info("Output: {}", str);
        String start = data.getWordVectorsTestData() + " ";
        Assert.assertTrue(str.startsWith(start));
        List<Double> res = Arrays.stream(str.replace(start, "").split("\\s+"))
                .mapToDouble(Double::parseDouble)
                .boxed()
                .collect(Collectors.toList());
        Assert.assertEquals(data.vecDim(), res.size());
        compareVectors(data.wordVectors(), res, data.vecDelta());
    }

    private static String runPrintVectors(Path bin, String testData, boolean sentence) throws Exception {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        InputStream _in = System.in;
        PrintStream _out = System.out;
        String cmd = sentence ? "print-sentence-vectors" : "print-word-vectors";
        LOGGER.debug("Run {}. Data '{}', bin: <{}>", cmd, testData, bin);
        try (PrintStream out = new PrintStream(output, true, StandardCharsets.UTF_8.name());
             InputStream in = new ByteArrayInputStream(testData.getBytes(StandardCharsets.UTF_8.name()))) {
            System.setIn(in);
            System.setOut(out);
            Main.run(cmd, bin.toString());
        } finally {
            System.setIn(_in);
            System.setOut(_out);
        }
        return new String(output.toByteArray(), StandardCharsets.UTF_8);
    }

    public enum Data {
        CBOX_THREAD4_DIM128_WS5_EPOCH10_MINCOUNT5 {
            @Override
            public String input() {
                return "/text-data.txt";
            }

            @Override
            public String args() {
                return "cbow -thread 4 -dim 128 -ws 5 -epoch 10 -minCount 5";
            }

            public String model() {
                return "junit.simple.cbox";
            }

            @Override
            public long binSize() {
                return 1_024_344_256;
            }

            @Override
            public long vecSize() {
                return 391_799;
            }

            @Override
            public int vecDim() {
                return 128;
            }

            @Override
            public int vecWords() {
                return 331;
            }

            @Override
            public List<Double> sentenceVectors() { // test sentence 'Word test.' 128-dim vector:
                return parse("0.11464 -0.21376 -0.095823 0.11768 0.0059741 -0.02797 -0.15613 0.02508 0.024723 -0.054983 " +
                        "-0.072046 0.11354 0.090018 -0.046901 0.097746 0.10175 -0.14776 0.04172 0.072863 0.017975 0.049459 " +
                        "-0.033836 -0.01571 -0.071825 0.021552 0.06303 0.020223 -0.021413 0.021902 -0.054924 0.12968 -0.097007 " +
                        "0.089375 -0.038077 0.053527 -0.068644 0.012673 -0.10477 -0.054977 -0.050762 -0.036415 -0.064909 -0.020371 " +
                        "0.14904 -0.086513 -0.067715 -0.011795 0.035342 0.10954 -0.064196 0.075644 0.044865 -0.047702 -0.094464 " +
                        "0.019276 -0.14534 -0.049696 0.0070982 0.070843 -0.014879 -0.039844 -0.14978 -0.020158 -0.17714 0.11351 " +
                        "-0.051109 -0.072486 -0.016916 -0.02824 -0.091959 0.12401 0.063377 0.038206 0.075664 -0.12715 0.0024533 " +
                        "-0.027761 -0.069692 0.060826 -0.083483 -0.020611 -0.17958 -0.015234 -0.067824 -0.050395 -0.017342 0.071854 " +
                        "0.055995 -0.14331 -0.14954 -0.019791 -0.0063819 0.15996 -0.062437 -0.078744 0.11477 0.097265 -0.029124 " +
                        "-0.25472 0.011654 -0.053102 0.061545 -0.10233 -0.070139 -0.076014 0.0022712 -0.031105 0.052113 0.090173 " +
                        "0.017779 0.066226 -0.078243 0.10734 0.01713 0.0029703 -0.10035 0.060397 0.043528 0.1954 0.022015 " +
                        "-0.0042344 -0.11538 -0.1566 0.011816 -0.2264 -0.039578 0.17037 -0.17236");
            }

            @Override
            public double vecDelta() {
                return 0.4f;
            }

            @Override
            public List<Double> wordVectors() { // test word 'Test'. 128-dim vector
                return parse("0.097585 -0.18108 -0.080698 0.10163 0.0055023 -0.024496 -0.13176 0.021189 0.020398 -0.050393 " +
                        "-0.060052 0.097613 0.077462 -0.040475 0.082243 0.08714 -0.12589 0.036291 0.061316 0.016788 0.040848 " +
                        "-0.026961 -0.01321 -0.060993 0.021236 0.056679 0.016582 -0.018349 0.019643 -0.046242 0.11014 -0.080717 " +
                        "0.074057 -0.031001 0.04629 -0.059101 0.0093017 -0.087812 -0.043051 -0.041892 -0.029324 -0.053696 " +
                        "-0.017312 0.12333 -0.069488 -0.060358 -0.01046 0.032383 0.089749 -0.057152 0.06475 0.039078 -0.041297 " +
                        "-0.080081 0.017931 -0.12257 -0.042619 0.0050177 0.061478 -0.012966 -0.033888 -0.12834 -0.015476 " +
                        "-0.15241 0.0953 -0.043704 -0.061819 -0.017165 -0.025126 -0.076223 0.10406 0.053756 0.032499 0.066904 " +
                        "-0.11043 0.00096502 -0.024105 -0.059445 0.051547 -0.071245 -0.017089 -0.15259 -0.012409 -0.057644 " +
                        "-0.044881 -0.013445 0.059415 0.044902 -0.11984 -0.12521 -0.018003 -0.004171 0.13584 -0.050801 -0.068329 " +
                        "0.096692 0.082484 -0.023193 -0.21677 0.010841 -0.045177 0.054996 -0.084878 -0.059781 -0.063631 0.0018137 " +
                        "-0.028321 0.041779 0.074388 0.016212 0.056248 -0.066612 0.09219 0.013714 0.0015402 -0.083682 0.04995 " +
                        "0.036039 0.16349 0.017082 -0.0020392 -0.096557 -0.1337 0.011869 -0.19198 -0.035029 0.14346 -0.14679");
            }

        },
        SUPERVISED_THREAD4_DIM10_LR01_NGRAMS2_BUCKET1E7_EPOCH5 {
            @Override
            public String input() {
                return "/dbpedia.cut.train";
            }

            @Override
            public String args() {
                return "supervised -dim 10 -lr 0.1 -wordNgrams 2 -minCount 1 -bucket 10000000 -epoch 5 -thread 4";
            }

            @Override
            public String model() {
                return "junit.simple.sup";
            }

            @Override
            public long binSize() {
                return 400_154_095;
            }

            @Override
            public long vecSize() {
                return 279_763;
            }

            @Override
            public int vecDim() {
                return 10;
            }

            @Override
            public int vecWords() {
                return 2695;
            }

            @Override
            public List<Double> wordVectors() { // dim: 10
                return Arrays.asList(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            }

            @Override
            public List<Double> sentenceVectors() {
                //-0.0092556 0.054957 -0.024542 -0.059048 0.0087722 0.04442 -0.0083089 0.033333 -0.0003131 0.032968
                //-0.0092685 0.054978 -0.024593 -0.059038 0.0087809 0.044432 -0.0082967 0.033355 -0.00029507 0.032974
                return parse("-0.0092685 0.054978 -0.024593 -0.059038 0.0087809 0.044432 -0.0082967 0.033355 -0.00029507 0.032974");
            }

            @Override
            public double vecDelta() {
                return 0.01;
            }
        },

        SKIPGRAM_THREAD3_DIM12_LR02_NGRAMS3_BUCKET5E6_EPOCH10_WS6 {
            @Override
            public String input() {
                return "/text-data.txt";
            }

            @Override
            public String args() {
                return "skipgram -thread 3 -dim 12 -lr 0.2 -wordNgrams 3 -bucket 5000000 -epoch 10 -ws 6";
            }

            @Override
            public String model() {
                return "junit.simple.sg";
            }

            @Override
            public long binSize() {
                return 240_037_088;
            }

            @Override
            public long vecSize() {
                return 35_577;
            }

            @Override
            public int vecDim() {
                return 12;
            }

            @Override
            public int vecWords() {
                return 331;
            }

            @Override
            public List<Double> wordVectors() { // for 'Test'
                return parse("0.024138 0.42543 -0.17895 0.1482 0.13226 0.048482 -0.22481 0.52733 0.052848 -0.0266 0.33456 0.06895");
            }

            @Override
            public List<Double> sentenceVectors() { // for 'Word test.'
                return parse("0.084481 0.5094 -0.1893 0.089596 0.2384 -0.086421 -0.1156 0.096683 0.29788 0.11489 0.096332 0.16538");
            }

            @Override
            public double vecDelta() {
                return 1;
            }
        },;

        public abstract String input();

        public abstract String args();

        public abstract String model();

        public abstract long binSize();

        public abstract long vecSize();

        public abstract int vecDim();

        public abstract int vecWords();

        public String getSentenceVectorsTestData() {
            return "Word test.";
        }

        public abstract List<Double> wordVectors();

        public String getWordVectorsTestData() {
            return "Test";
        }

        public abstract List<Double> sentenceVectors();

        public abstract double vecDelta();

        public Path getInput() throws URISyntaxException, IOException {
            return Paths.get(Data.class.getResource(input()).toURI()).toRealPath();
        }

        public Path getOutput() {
            return DESTINATION_DIR.resolve(model());
        }

        public Path getModelBin() {
            return Paths.get(getOutput().toString() + ".bin");
        }

        public Path getModelVec() {
            return Paths.get(getOutput().toString() + ".vec");
        }

        protected static List<Double> parse(String data) {
            return Arrays.stream(data.split("\\s+")).map(Double::parseDouble).collect(Collectors.toList());
        }

        public String[] command() throws Exception {
            return TestsBase.cmd(args() + " -input %s -output %s", getInput(), getOutput());
        }
    }

    private static class Word implements Comparable<Word> {
        String word;
        List<Float> vec;
        String in;

        private Word(String line) {
            String[] arr = (in = line).split("\\s");
            word = arr[0];
            vec = Arrays.stream(arr).skip(1).map(Float::parseFloat).collect(Collectors.toList());
        }

        @Override
        public String toString() {
            return String.format("%s:::%s}", word, vec);
        }

        @Override
        public int compareTo(Word o) {
            return word.compareTo(o.word);
        }
    }

    private static Set<String> toSet(List<Word> in) {
        return in.stream().map(w -> w.word).collect(Collectors.toSet());
    }

    private static Map<String, List<Float>> toMap(List<Word> in) {
        return in.stream().collect(Collectors.toMap(word -> word.word, word -> word.vec));
    }

    private static List<Word> collect(Path path) throws IOException {
        try (BufferedReader r = Files.newBufferedReader(path)) {
            return r.lines().skip(1).map(Word::new).collect(Collectors.toList());
        }
    }

}
