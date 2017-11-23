package ru.avicomp;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import org.junit.Assert;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ru.avicomp.tests.SimpleModelTest;

/**
 * Tests data and result dir.
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

    public static String[] cmd(SimpleModelTest.Data data) throws IOException, URISyntaxException {
        return cmd(data.cmd(), data.getInput(), data.getOutput());
    }

    public static void compareVectors(List<Double> expected, List<Double> actual, double delta) {
        LOGGER.debug("Expected {}", expected);
        LOGGER.debug("Actual {}", actual);

        Assert.assertEquals("Wrong vectors size", expected.size(), actual.size());
        for (int i = 0; i < expected.size(); i++) {
            Assert.assertEquals("#" + i, expected.get(i), actual.get(i), delta);
        }
    }
}
