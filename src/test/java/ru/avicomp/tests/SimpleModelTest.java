package ru.avicomp.tests;

import java.io.*;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

import org.junit.Assert;
import org.junit.FixMethodOrder;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.MethodSorters;
import org.junit.runners.Parameterized;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.fasttext.Args;
import cc.fasttext.Main;
import ru.avicomp.TestsBase;

import static ru.avicomp.TestsBase.*;

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
        Main.train(cmd(data));

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
        List<String> res = Arrays.stream(array.toString(StandardCharsets.UTF_8.name()).split("\r*\n")).collect(Collectors.toList());
        res.stream()
                .limit(10)
                .forEach(s -> LOGGER.debug("{}", s));
        LOGGER.info("Size: {}", res.size());
        Assert.assertEquals("Wrong count of lines in out", 161, res.size());
        Assert.assertEquals("Wrong first label", Args.DEFAULT_LABEL + 12, res.get(0));
        Assert.assertEquals("Should be one unique label", 1, new HashSet<>(res).size());
    }

    @Test
    public void test04PrintSentenceVectors() throws Exception {
        String str = runPrintVectors(data.getModelBin(), data.getSentenceVectorsTestData(), true);
        LOGGER.info("Output: {}", str);
        List<Double> res = Arrays.stream(str.split("\\s+"))
                .mapToDouble(Double::parseDouble)
                .boxed()
                .collect(Collectors.toList());
        compareVectors(data.wordVectors(), res, 0.4);
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
        compareVectors(data.sentenceVectors(), res, 0.4);
    }

    private static String runPrintVectors(Path bin, String testData, boolean sentence) throws Exception {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        InputStream _in = System.in;
        PrintStream _out = System.out;
        String cmd = sentence ? "print-sentence-vectors" : "print-word-vectors";
        LOGGER.debug("Run " + cmd + ". Data '" + testData + "'");
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
            public String cmd() {
                return "cbow -thread 4 -dim 128 -ws 5 -epoch 10 -minCount 5 -input %s -output %s";
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

            public String model() {
                return "junit.cbox.t4.d128.w5.e10.m5";
            }

            @Override
            public List<Double> wordVectors() { // 128 vectors:
                return Arrays.asList(-0.01188, 0.0044223, -0.066131, -0.020825, -0.14696, -0.099169, 0.11134, -0.11649,
                        -0.045337, 0.063723, -0.029514, -0.090043, -0.092293, 0.02329, -0.042874, 0.17592, -0.10492,
                        -0.14772, 0.023096, 0.04982, 0.10333, -0.019804, -0.18756, 0.15125, -0.10442, -0.077873,
                        0.034354, 0.093182, 0.16364, 0.014039, -0.055895, -0.060817, -0.016289, -0.023442, 0.1632,
                        -0.1463, 0.14701, 0.017004, 0.038988, 0.018109, -0.12639, -0.0090438, 0.10515, -0.1256, 0.060793,
                        -0.030233, 0.026761, 0.123, 0.043514, 0.0076771, -0.046018, 0.0014352, 0.002032, -0.18788,
                        -0.15562, -0.084914, 0.057617, 0.040016, 0.11375, 0.1164, 0.11189, 0.069087, 0.078974,
                        -0.040239, 0.030464, 0.11324, -0.032604, 0.14373, 0.039304, 0.06836, -0.045781, -0.095826,
                        -0.081032, 0.12145, 0.069719, 0.11913, 0.0089484, -0.066021, -0.13351, 0.14071, -0.18872, 0.01186,
                        -0.12922, 0.097781, -0.087912, -0.003281, -0.059217, 0.075035, -0.18873, -0.047837, 0.026048,
                        -0.039929, 0.014265, -0.052473, -0.026612, 0.09175, 0.078742, -0.040234, -0.023985, 0.097959,
                        0.080668, 0.11665, 0.043819, 0.021178, 0.070436, -0.017433, 0.0033352, 0.015667, -0.10719,
                        -0.052713, 0.084073, -0.15568, -0.043047, -0.044119, -0.035038, 0.0064535, -0.035223, 0.22574,
                        -0.056209, 0.12817, -0.016375, 0.041407, -0.027459, -0.019981, -0.035947, -0.058359, 0.060406, -0.072145);
            }

            @Override
            public List<Double> sentenceVectors() {
                return Arrays.asList(-0.0089522, 0.0039847, -0.046722, -0.01396, -0.10603, -0.072662, 0.081412, -0.085983,
                        -0.032709, 0.045654, -0.021025, -0.066668, -0.065771, 0.018684, -0.032078, 0.13132, -0.076479,
                        -0.10762, 0.015852, 0.035983, 0.075114, -0.014991, -0.13578, 0.11287, -0.076567, -0.057383,
                        0.025914, 0.067416, 0.11948, 0.010497, -0.043517, -0.044062, -0.012366, -0.016839, 0.11769,
                        -0.10937, 0.10655, 0.0094078, 0.028129, 0.012603, -0.093993, -0.0047002, 0.076851, -0.089753,
                        0.041172, -0.02051, 0.017041, 0.09011, 0.032132, 0.0062371, -0.033948, 0.0015065, 0.0010718,
                        -0.1363, -0.11411, -0.065492, 0.040423, 0.03084, 0.08206, 0.083906, 0.082571, 0.051587, 0.05973,
                        -0.029015, 0.023743, 0.084154, -0.021363, 0.10646, 0.030054, 0.049582, -0.034118, -0.068586,
                        -0.058226, 0.089597, 0.051972, 0.087976, 0.0043332, -0.047903, -0.0974, 0.10439, -0.13827,
                        0.0089531, -0.094378, 0.071923, -0.066227, -5.6161E-4, -0.046614, 0.051582, -0.14075, -0.034901,
                        0.017811, -0.028764, 0.0094627, -0.039578, -0.020295, 0.067079, 0.05873, -0.02924, -0.019213,
                        0.072415, 0.05911, 0.085429, 0.033362, 0.014874, 0.050846, -0.013746, 0.0032173, 0.0086438,
                        -0.076945, -0.038504, 0.063941, -0.11354, -0.033476, -0.035254, -0.0267, 0.0054636, -0.026594,
                        0.16536, -0.042795, 0.094596, -0.011233, 0.031857, -0.017835, -0.014792, -0.025392, -0.042061,
                        0.045974, -0.054442);
            }
        },
        SUPERVISED_THREAD4_DIM10_LR01_NGRAMS2_BUCKET1E7_EPOCH5 {
            @Override
            public String input() {
                return "/dbpedia.cut.train";
            }

            @Override
            public String cmd() {
                return "supervised -dim 10 -lr 0.1 -wordNgrams 2 -minCount 1 -bucket 10000000 -epoch 5 -thread 4 -input %s -output %s";
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
            public String model() {
                return "junit.sup.t4.d10.lr01.wn2.b1e7.e5.m1";
            }

            @Override
            public List<Double> sentenceVectors() { // 10
                return Arrays.asList(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            }

            @Override
            public List<Double> wordVectors() {
                return Arrays.asList(0.0079745, -0.014468, -0.071323, 0.054646, 0.037028, 0.070605, -0.010022, 0.082145, 0.018551, -0.002894);
            }
        };

        public abstract String input();

        public abstract String cmd();

        public abstract long binSize();

        public abstract long vecSize();

        public abstract int vecDim();

        public abstract int vecWords();

        public abstract String model();

        public String getSentenceVectorsTestData() {
            return "Word test.";
        }

        public abstract List<Double> sentenceVectors();

        public String getWordVectorsTestData() {
            return "Test";
        }

        public abstract List<Double> wordVectors();

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
