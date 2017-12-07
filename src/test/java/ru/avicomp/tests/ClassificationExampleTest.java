package ru.avicomp.tests;

import cc.fasttext.FastText;
import cc.fasttext.Main;
import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.FixMethodOrder;
import org.junit.Test;
import org.junit.runners.MethodSorters;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import ru.avicomp.ShellUtils;
import ru.avicomp.TestsBase;

import java.io.*;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Based on
 * <a href='https://github.com/facebookresearch/fastText/blob/master/classification-example.sh'>classification-example.sh</a> and
 * <a href='https://github.com/facebookresearch/fastText/blob/master/quantization-example.sh'>quantization-example.sh</a>
 * scripts.
 * <p>
 * Created by @szuev on 01.11.2017.
 */
@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class ClassificationExampleTest {
    private static final Logger LOGGER = LoggerFactory.getLogger(ClassificationExampleTest.class);

    // official data:
    private static final String DBPEDIA_TAR_GZ_URL = "https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz";
    private static final String DBPEDIA_TAR_GZ_FILE = "dbpedia_csv.tar.gz";
    private static final String DBPEDIA_DIR = "dbpedia_csv";
    private static final String DBPEDIA_TRAN_CSV = "train.csv";
    private static final String DBPEDIA_TEST_CSV = "test.csv";

    private static final String DBPEDIA_TRAN = "dbpedia.train";
    private static final String DBPEDIA_TEST = "dbpedia.test";
    private static final String DBPEDIA_MODEL = "dbpedia";

    // expected size of model for dbpedia_csv/train.csv file:
    private static final long DBPEDIA_MODEL_BIN_SIZE = 447_481_878;
    // 1_623_284
    private static final long DBPEDIA_MODEL_FTZ_SIZE = 1_623_778;

    private static Path train, test, model;

    @BeforeClass
    public static void before() throws IOException {
        LOGGER.info("Preparation");
        Path dir = TestsBase.DESTINATION_DIR.resolve(DBPEDIA_DIR);
        train = TestsBase.DESTINATION_DIR.resolve(DBPEDIA_TRAN);
        test = TestsBase.DESTINATION_DIR.resolve(DBPEDIA_TEST);
        model = TestsBase.DESTINATION_DIR.resolve(DBPEDIA_MODEL);

        if (Files.exists(train) && Files.exists(test)) {
            return;
        }

        Path trainCSV = dir.resolve(DBPEDIA_TRAN_CSV);
        Path testCSV = dir.resolve(DBPEDIA_TEST_CSV);
        if (!Files.exists(testCSV) || !Files.exists(trainCSV)) {
            Path archive = TestsBase.DESTINATION_DIR.resolve(DBPEDIA_TAR_GZ_FILE);
            if (!Files.exists(archive)) {
                ShellUtils.download(new URL(DBPEDIA_TAR_GZ_URL), archive);
            }
            ShellUtils.unpackTarGZ(archive, TestsBase.DESTINATION_DIR);
        }
        ShellUtils.normalizeAndShuffle(trainCSV, train);
        ShellUtils.normalizeAndShuffle(testCSV, test);
    }

    @Test
    public void test01TrainModel() throws Exception {
        Path bin = TestsBase.DESTINATION_DIR.resolve(DBPEDIA_MODEL + ".bin");
        Path vec = TestsBase.DESTINATION_DIR.resolve(DBPEDIA_MODEL + ".vec");
        Path out = TestsBase.DESTINATION_DIR.resolve(DBPEDIA_MODEL + ".output");
        Main.train(TestsBase.cmd("supervised" +
                " -input %s" +
                " -output %s" +
                " -dim 10" +
                " -lr 0.1" +
                " -wordNgrams 2" +
                " -minCount 1" +
                " -bucket 10000000" +
                " -epoch 5" +
                " -thread 4" +
                " -saveOutput 555", train, model));
        Assert.assertTrue("No .bin found", Files.exists(bin));
        Assert.assertTrue("No .vec found", Files.exists(vec));
        Assert.assertTrue("No .output found", Files.exists(out));
        Assert.assertEquals("Incorrect size of dbpedia.bin model", DBPEDIA_MODEL_BIN_SIZE, Files.size(bin));
        // todo: validate .output and .vec
    }

    @Test
    public void test02QuantizeModel() throws Exception {
        Path model = TestsBase.DESTINATION_DIR.resolve(DBPEDIA_MODEL);
        Path ftz = TestsBase.DESTINATION_DIR.resolve(DBPEDIA_MODEL + ".ftz");
        Path vec = TestsBase.DESTINATION_DIR.resolve(DBPEDIA_MODEL + ".vec");
        Main.quantize(TestsBase.cmd("quantize" +
                " -input %s" +
                " -output %s" +
                " -qnorm" +
                " -retrain" +
                " -epoch 1" +
                " -cutoff 100000", train, model));
        Assert.assertTrue("No .ftz found", Files.exists(ftz));
        Assert.assertTrue("No .vec found", Files.exists(vec));
        // 0.5 kb allowed diff:
        Assert.assertEquals("Incorrect size of dbpedia.ftz model", DBPEDIA_MODEL_FTZ_SIZE, Files.size(ftz), 500);
    }

    private Path getModelBinPath() throws Exception {
        Path res = TestsBase.DESTINATION_DIR.resolve(DBPEDIA_MODEL + ".bin");
        if (!Files.exists(res)) {
            test01TrainModel();
        }
        return res;
    }

    private Path getModelFtzPath() throws Exception {
        Path res = TestsBase.DESTINATION_DIR.resolve(DBPEDIA_MODEL + ".ftz");
        if (!Files.exists(res)) {
            test02QuantizeModel();
        }
        return res;
    }

    @Test
    public void test03TestBin() throws Exception {
        testTest(getModelBinPath(), false);
    }

    @Test
    public void test04TestVtz() throws Exception { // todo: fix
        testTest(getModelFtzPath(), true);
    }

    private void testTest(Path model, boolean quant) throws IOException {
        // output:
        // N	70000
        // P@2: 0,495
        // R@2: 0,990
        // Number of examples: 70000
        int expectedN = 70_000;
        double expectedR = 1;
        double expectedP = 0.5;
        double delta = 0.05;
        int k = 2;
        LOGGER.info("Test file {}, Model.bin {}", test, model);
        FastText f = Main.loadModel(model.toString());
        Assert.assertEquals(quant, f.getModel().isQuant());
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        try (InputStream in = Files.newInputStream(test);
             PrintStream out = new PrintStream(output, true, StandardCharsets.UTF_8.name())) {
            f.test(in, out, k);
        }
        String res = new String(output.toByteArray(), StandardCharsets.UTF_8);
        LOGGER.debug("Output: {}", res);
        String[] lines = res.split("\n");
        Assert.assertEquals(4, lines.length);
        int n = Integer.parseInt(lines[0].replaceAll("[^\\d]", ""));
        double p = Double.parseDouble(lines[1].split("\\s+")[1]);
        double r = Double.parseDouble(lines[2].split("\\s+")[1]);
        Assert.assertEquals("Wrong number of examples", expectedN, n);
        Assert.assertEquals("Wrong P@" + k, expectedP, p, delta);
        Assert.assertEquals("Wrong R@" + k, expectedR, r, delta);
    }

    @Test
    public void test05PredictFile() throws Exception {
        LOGGER.info("Test prediction");
        Path result = Paths.get(ClassificationExampleTest.class.getResource("/dbpedia.cut.test.predict").toURI());
        Path test = Paths.get(ClassificationExampleTest.class.getResource("/dbpedia.cut.test").toURI());
        List<String> expected = Files.lines(result)
                .map(String::trim)
                .collect(Collectors.toList());


        String bin = getModelBinPath().toString();
        FastText f = Main.loadModel(bin);

        List<String> actual;
        LOGGER.info("predict {} {} 1", bin, test);
        try (ByteArrayOutputStream array = new ByteArrayOutputStream();
             PrintStream out = new PrintStream(array);
             InputStream in = Files.newInputStream(test)) {
            f.setLogs(out);
            f.predict(in, 1, false);
            actual = Arrays.stream(array.toString(StandardCharsets.UTF_8.name()).split("\n"))
                    .map(String::trim)
                    .collect(Collectors.toList());
        }
        Assert.assertEquals(expected.size(), actual.size());
        Assert.assertEquals(expected, actual);
    }

    @Test
    public void test06PredictProbStdIn() throws Exception {
        LOGGER.info("Test predict with probabilities");
        Path model = getModelBinPath();
        List<String> testData = Arrays.asList("predict", "test");
        List<Map<String, List<Float>>> res7 = predictProb(model, testData, 7);
        List<Map<String, List<Float>>> res5 = predictProb(model, testData, 5);

        Assert.assertEquals("Wrong results", res7.size(), res5.size());
        for (int i = 0; i < res7.size(); i++) {
            Map<String, List<Float>> map7 = res7.get(i);
            Map<String, List<Float>> map5 = res5.get(i);
            Assert.assertTrue("Can't find some labels", map7.keySet().containsAll(map5.keySet()));
            for (String label : map5.keySet()) {
                List<Float> floats7 = map7.get(label);
                List<Float> floats5 = map5.get(label);
                Assert.assertTrue("the values do not match", floats7.containsAll(floats5));
            }
        }
    }

    private static List<Map<String, List<Float>>> predictProb(Path bin, List<String> testData, int k) throws Exception {
        String input = testData.stream().collect(Collectors.joining("\n")) + "\n";
        LOGGER.info("Input: {}", input);

        ByteArrayOutputStream output = new ByteArrayOutputStream();
        InputStream _in = System.in;
        PrintStream _out = System.out;
        try (PrintStream out = new PrintStream(output, true, StandardCharsets.UTF_8.name());
             InputStream in = new ByteArrayInputStream(input.getBytes(StandardCharsets.UTF_8.name()))) {
            System.setIn(in);
            System.setOut(out);
            Main.run(TestsBase.cmd("predict-prob %s %s " + k, bin, "-"));
        } finally {
            System.setIn(_in);
            System.setOut(_out);
        }
        String str = new String(output.toByteArray(), StandardCharsets.UTF_8);
        LOGGER.info("Output: {}", str);

        List<Map<String, List<Float>>> res = new ArrayList<>();
        String[] lines = str.split("\n");
        Assert.assertEquals("Wrong lines", testData.size(), lines.length);
        for (int i = 0; i < lines.length; i++) {
            Map<String, List<Float>> map = parsePredictLine(lines[i]);
            LOGGER.debug("'{}'\t=>\t{}", testData.get(i), map);
            int size = map.values().stream().mapToInt(List::size).sum();
            Assert.assertEquals("Wrong labels count", k, size);
            res.add(map);
        }
        return res;
    }

    private static Map<String, List<Float>> parsePredictLine(String line) {
        String[] array = line.split("\\s");
        Assert.assertEquals("Wrong line '" + line + "'", 0, array.length % 2);
        Map<String, List<Float>> res = new HashMap<>();
        for (int i = 0; i < array.length; i++) {
            String label = array[i];
            Float value = Float.valueOf(array[++i]);
            res.computeIfAbsent(label, s -> new ArrayList<>()).add(value);
        }
        return res;
    }

}

