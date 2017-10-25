package ru.avicomp;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Created by @szuev on 24.10.2017.
 */
public class TestsBase {
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

    public static String[] cmd(String cmd, Path in, Path out) {
        String res = String.format(cmd, in, out);
        System.out.println("Cmd: " + res);
        return res.split("\\s");
    }
}
