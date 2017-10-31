package ru.avicomp;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.junit.BeforeClass;
import org.junit.Test;

import cc.fasttext.Main;

/**
 * Created by @szuev on 31.10.2017.
 */
public class PredictTest {

    private static TestsBase.Data data = TestsBase.Data.SUPERVISED_THREAD4_DIM10_LR01_NGRAMS2_BUCKET1E7_EPOCH5;

    @BeforeClass
    public static void createModel() throws Exception {
        if (Files.exists(data.getModelBin())) return;
        new Main().train(TestsBase.cmd(data));
    }

    @Test
    public void test() throws Exception {
        String cmd = "predict %s %s";
        Path in = Paths.get(PredictTest.class.getResource("/dbpedia.cut.test").toURI());

        ByteArrayOutputStream array = new ByteArrayOutputStream();
        PrintStream newOut = new PrintStream(array);
        PrintStream out = System.out;
        try {
            System.setOut(newOut);
            new Main().predict(TestsBase.cmd(cmd, data.getModelBin(), in));
        } finally {
            System.setOut(out);
        }
        List<String> res = Arrays.stream(array.toString(StandardCharsets.UTF_8.name()).split("\n")).collect(Collectors.toList());
        res.stream()
                .distinct().limit(10)
                .forEach(s -> TestsBase.LOGGER.debug("{}", s));
        TestsBase.LOGGER.info("Size: {}", res.size());
    }
}
