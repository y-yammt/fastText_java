package cc.fasttext.io;

import java.util.Locale;

import org.apache.commons.lang.Validate;

/**
 * Helper to print
 * Created by @szuev on 13.12.2017.
 */
public class FormatUtils {

    /**
     * Default: should be similar to c++ output for float
     *
     * @param number float
     * @return String
     */
    public static String toString(float number) {
        return toString(number, 5);
    }

    /**
     * Formats a float value in c++ linux style
     *
     * @param number    float
     * @param precision int, positive
     * @return String
     */
    public static String toString(float number, int precision) {
        Validate.isTrue(precision > 0);
        return String.format(Locale.US, "%." + precision + "g", number).replaceFirst("0+($|e)", "$1").replaceFirst("\\.$", "");
    }
}
