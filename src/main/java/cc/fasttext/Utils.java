package cc.fasttext;

import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
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


    public static byte[] readUpToByte(InputStream in, byte end) throws IOException {
        List<Integer> buff = new ArrayList<>(128);
        while (true) {
            int c = in.read();
            if (c == end) {
                break;
            }
            if (c == -1) throw new EOFException();
            buff.add(c);
        }
        byte[] res = new byte[buff.size()];
        for (int i = 0; i < buff.size(); i++) {
            res[i] = buff.get(i).byteValue();
        }
        return res;
    }

    public static void writeString(OutputStream out, String s) throws IOException {
        out.write(s.getBytes());
        out.write(0);
    }

    public static String readString(InputStream in) throws IOException {
        return new String(readUpToByte(in, (byte) 0), StandardCharsets.UTF_8);
    }

    public static String formatNumber(double d) {
        return String.format(Locale.US, "%g", d).replaceAll("0+($|e)", "$1");
    }
}
