package ru.avicomp.tests;

import java.io.*;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

import org.junit.Assert;
import org.junit.BeforeClass;
import org.junit.FixMethodOrder;
import org.junit.Test;
import org.junit.runners.MethodSorters;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.fasttext.Args;
import cc.fasttext.FastText;
import cc.fasttext.Main;
import ru.avicomp.ShellUtils;
import ru.avicomp.TestsBase;

/**
 * Original script:
 * <p>
 * <pre>{@code
 * #!/usr/bin/env bash
 * myshuf() {
 *  perl -MList::Util=shuffle -e 'print shuffle(<>);' "$@";
 * }
 * normalize_text() {
 *  tr '[:upper:]' '[:lower:]' | sed -e 's/^/__label__/g' | \
 *  sed -e "s/'/ ' /g" -e 's/"//g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' \
 *  -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
 *  -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' | tr -s " " | myshuf
 * }
 * RESULTDIR=result
 * DATADIR=data
 * mkdir -p "${RESULTDIR}"
 * mkdir -p "${DATADIR}"
 * if [ ! -f "${DATADIR}/dbpedia.train" ]
 * then
 *  wget -c "https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz" -O "${DATADIR}/dbpedia_csv.tar.gz"
 *  tar -xzvf "${DATADIR}/dbpedia_csv.tar.gz" -C "${DATADIR}"
 *  cat "${DATADIR}/dbpedia_csv/train.csv" | normalize_text > "${DATADIR}/dbpedia.train"
 *  cat "${DATADIR}/dbpedia_csv/test.csv" | normalize_text > "${DATADIR}/dbpedia.test"
 * fi
 * make
 * ./fasttext supervised -input "${DATADIR}/dbpedia.train" -output "${RESULTDIR}/dbpedia" -dim 10 -lr 0.1 -wordNgrams 2 -minCount 1 -bucket 10000000 -epoch 5 -thread 4
 * ./fasttext test "${RESULTDIR}/dbpedia.bin" "${DATADIR}/dbpedia.test"
 * ./fasttext predict "${RESULTDIR}/dbpedia.bin" "${DATADIR}/dbpedia.test" > "${RESULTDIR}/dbpedia.test.predict"</pre>
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
        Args args = Main.parseArgs(TestsBase.cmd("supervised -input %s -output %s -dim 10 -lr 0.1 " +
                "-wordNgrams 2 -minCount 1 -bucket 10000000 -epoch 5 -thread 4", train, model));
        new FastText(args).train();
        Assert.assertEquals("Incorrect size of dbpedia model", DBPEDIA_MODEL_BIN_SIZE, Files.size(bin));
    }

    private Path getModelBinPath() throws Exception {
        Path bin = TestsBase.DESTINATION_DIR.resolve(DBPEDIA_MODEL + ".bin");
        if (!Files.exists(bin)) {
            test01TrainModel();
        }
        return bin;
    }

    @Test
    public void test03PredictFile() throws Exception {
        LOGGER.info("Test prediction");
        Path result = Paths.get(ClassificationExampleTest.class.getResource("/dbpedia.cut.test.predict").toURI());
        Path test = Paths.get(ClassificationExampleTest.class.getResource("/dbpedia.cut.test").toURI());
        List<String> expected = Files.lines(result)
                .map(String::trim)
                .collect(Collectors.toList());

        FastText f = new FastText(Main.createArgs());
        String bin = getModelBinPath().toString();
        f.loadModel(bin);
        List<String> actual;
        LOGGER.info("predict {} {} 1", bin, test);
        try (ByteArrayOutputStream array = new ByteArrayOutputStream();
             PrintStream out = new PrintStream(array);
             InputStream in = Files.newInputStream(test)) {
            f.setPrintOut(out);
            f.predict(in, 1, false);
            actual = Arrays.stream(array.toString(StandardCharsets.UTF_8.name()).split("\n"))
                    .map(String::trim)
                    .collect(Collectors.toList());
        }
        Assert.assertEquals(expected.size(), actual.size());
        Assert.assertEquals(expected, actual);
    }

    @Test
    public void test03PredictProbStdIn() throws Exception {
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

