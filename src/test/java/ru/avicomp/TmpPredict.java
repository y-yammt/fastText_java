package ru.avicomp;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.junit.Assert;
import org.junit.Test;

import cc.fasttext.Main;

/**
 * TODO: remove
 * Created by @szuev on 01.11.2017.
 */
public class TmpPredict {

    /*public static void main(String ... args) throws Exception {
        Path root = Paths.get("../");
        testPredict(root.resolve("dbpedia.bin").toRealPath(), root.resolve("dbpedia.test").toRealPath());
        testPredict(root.resolve("dbpedia.cut.model.bin").toRealPath(), root.resolve("dbpedia.test.cut").toRealPath());
    }*/

    @Test
    public void testDBPediaPredict() throws Exception {
        Path root = Paths.get("../");
        List<String> actual = predict(root.resolve("dbpedia.bin").toRealPath(), root.resolve("dbpedia.test").toRealPath());
        List<String> expected = Files.lines(root.resolve("dbpedia.test.predict").toRealPath())
                .map(String::trim).collect(Collectors.toList());
        Assert.assertEquals("wrong size", expected.size(), actual.size());
        List<String> errors = new ArrayList<>();
        for (int i = 0; i < expected.size(); i++) {
            String e = expected.get(i);
            String a = actual.get(i);
            if (e.equals(a)) continue;
            errors.add(i + ":::" + e + " != " + a);
        }
        errors.forEach(TestsBase.LOGGER::error);
        Assert.assertTrue("Errors " + errors.size(), errors.isEmpty());
    }

    public static void testPredict(Path bin, Path test) throws Exception {
        TestsBase.LOGGER.info("Test Predict {} {}", bin, test);
        predict(bin, test).stream()
                .limit(10)
                .forEach(s -> TestsBase.LOGGER.debug("{}", s));
    }

    public static List<String> predict(Path bin, Path test) throws Exception {
        String[] cmd = TestsBase.cmd("predict %s %s", bin, test);
        ByteArrayOutputStream array = new ByteArrayOutputStream();
        PrintStream newOut = new PrintStream(array);
        PrintStream out = System.out;
        try {
            System.setOut(newOut);
            new Main().predict(cmd);
        } finally {
            System.setOut(out);
        }
        return Arrays.stream(array.toString(StandardCharsets.UTF_8.name()).split("\n")).map(String::trim).collect(Collectors.toList());
    }
}

