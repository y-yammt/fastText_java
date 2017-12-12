package ru.avicomp;

import org.junit.Assert;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Tests helper (data and result dir).
 *
 * Created by @szuev on 24.10.2017.
 */
public final class TestsBase {
    public static final Logger LOGGER = LoggerFactory.getLogger(TestsBase.class);

    public static final Path DESTINATION_DIR = Paths.get("out");

    static {
        init();
    }

    private static void init() {
        try {
            Files.createDirectories(DESTINATION_DIR);
        } catch (IOException e) {
            throw new AssertionError(e);
        }
    }

    public static String[] cmd(String cmd, Object in, Object out) {
        String res = String.format(cmd, in, out);
        LOGGER.info("Cmd: {}", res);
        return res.split("\\s");
    }

    public static void compareVectors(List<Double> expected, List<Double> actual, double delta) {
        LOGGER.debug("Expected {}", expected);
        LOGGER.debug("Actual {}", actual);

        Assert.assertEquals("Wrong vectors size", expected.size(), actual.size());
        for (int i = 0; i < expected.size(); i++) {
            Assert.assertEquals("#" + i, expected.get(i), actual.get(i), delta);
        }
    }

    public static <T> void compareLists(List<T> expected, List<T> actual, int allowedDeviation) {
        Assert.assertEquals(expected.size(), actual.size());
        LOGGER.debug("E: {}", expected);
        LOGGER.debug("A: {}", actual);
        List<String> errors = new ArrayList<>();
        for (int i = 0; i < actual.size(); i++) {
            if (expected.get(i).equals(actual.get(i))) continue;
            errors.add(String.format("Wrong label #%d: expected('%s')!=actual('%s')", i, expected.get(i), actual.get(i)));
        }
        errors.forEach(LOGGER::warn);
        Assert.assertTrue("Errors: " + errors.size(), errors.size() <= allowedDeviation);
    }
}
