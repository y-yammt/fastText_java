package cc.fasttext.io;

import org.apache.commons.lang.StringUtils;

import java.io.PrintStream;
import java.util.Formatter;
import java.util.Locale;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * Print logs while operations with {@link cc.fasttext.FastText FastText}.
 * Analogues of {@link java.io.PrintStream}
 * <p>
 * Created by @szuev on 08.12.2017.
 */
public interface PrintLogs {
    PrintLogs NULL = new Fake();
    PrintLogs STANDARD = new Stream(System.out);

    void print(String str);

    void println(String str);

    void println();

    void printf(String format, Object... args);

    /**
     * Empty impl of {@link PrintLogs}
     */
    class Fake implements PrintLogs {
        @Override
        public void print(String str) {

        }

        @Override
        public void println(String str) {

        }

        @Override
        public void println() {

        }

        @Override
        public void printf(String format, Object... args) {

        }
    }

    /**
     * To print to console, adapter for {@link PrintStream}
     */
    class Stream implements PrintLogs {

        private final PrintStream out;

        public Stream(PrintStream out) {
            this.out = Objects.requireNonNull(out, "Null out");
        }

        @Override
        public void print(String str) {
            out.print(str);
        }

        @Override
        public void println(String str) {
            out.println(str);
        }

        @Override
        public void println() {
            out.println();
        }

        @Override
        public void printf(String format, Object... args) {
            out.printf(format, args);
        }
    }

    /**
     * To use with java logger.
     */
    class Flat implements PrintLogs {
        private final Consumer<String> logger;
        private final Locale locale;

        public Flat(Locale locale, Consumer<String> logger) {
            this.logger = Objects.requireNonNull(logger, "Null logger.");
            this.locale = locale;
        }

        public Flat(Consumer<String> logger) {
            this(Locale.US, logger);
        }

        @Override
        public void print(String str) {
            logger.accept(prepare(str));
        }

        @Override
        public void println(String str) {
            logger.accept(prepare(str));
        }

        @Override
        public void println() {

        }

        @Override
        public void printf(String format, Object... args) {
            logger.accept(new Formatter(locale).format(prepare(format), args).toString());
        }

        private static String prepare(String msg) {
            return StringUtils.replaceChars(msg, "\r\n", null);
        }
    }
}
