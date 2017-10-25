package cc.fasttext;

import java.io.*;
import java.util.Map;
import java.util.Random;

public class Utils {

    /**
     * Ensures the truth of an expression involving one or more parameters to
     * the calling method.
     *
     * @param expression a boolean expression
     * @throws IllegalArgumentException if {@code expression} is false
     */
    public static void checkArgument(boolean expression) {
        if (!expression) {
            throw new IllegalArgumentException();
        }
    }

    public static void checkArgument(boolean expression, String message) {
        if (!expression) {
            throw new IllegalArgumentException(message);
        }
    }

    public static boolean isEmpty(String str) {
        return (str == null || str.isEmpty());
    }

    public static <K, V> V mapGetOrDefault(Map<K, V> map, K key, V defaultValue) {
        return map.getOrDefault(key, defaultValue);
    }

    public static int randomInt(Random rnd, int lower, int upper) {
        checkArgument(lower <= upper & lower > 0);
        if (lower == upper) {
            return lower;
        }
        return rnd.nextInt(upper - lower) + lower;
    }

    /**
     * The snippet
     * <pre>{@code
     * std::minstd_rand rng(seed));
     * std::uniform_int_distribution<> uniform(start, end);
     * }</pre>
     * generates values from interval [start, end]!
     *
     * @param rnd
     * @param start, inclusive
     * @param end,   inclusive
     * @return
     */
    public static int nextInt(Random rnd, int start, int end) {
        checkArgument(start >= 0);
        checkArgument(start <= end);
        return rnd.nextInt(end - start + 1) + start;
    }

    public static float randomFloat(Random rnd, float lower, float upper) {
        checkArgument(lower <= upper);
        if (lower == upper) {
            return lower;
        }
        return (rnd.nextFloat() * (upper - lower)) + lower;
    }

    public static long sizeLine(String filename) throws IOException {
        InputStream is = new BufferedInputStream(new FileInputStream(filename));
        try {
            byte[] c = new byte[1024];
            long count = 0;
            int readChars = 0;
            boolean endsWithoutNewLine = false;
            while ((readChars = is.read(c)) != -1) {
                for (int i = 0; i < readChars; ++i) {
                    if (c[i] == '\n')
                        ++count;
                }
                endsWithoutNewLine = (c[readChars - 1] != '\n');
            }
            if (endsWithoutNewLine) {
                ++count;
            }
            return count;
        } finally {
            is.close();
        }
    }

    /**
     * @param br
     * @param pos line numbers start from 1
     * @throws IOException
     */
    public static void seekLine(BufferedReader br, long pos) throws IOException {
        // br.reset();
        String line;
        int currentLine = 1;
        while (currentLine < pos && (line = br.readLine()) != null) {
            if (Utils.isEmpty(line) || line.startsWith("#")) {
                continue;
            }
            currentLine++;
        }
    }

}
