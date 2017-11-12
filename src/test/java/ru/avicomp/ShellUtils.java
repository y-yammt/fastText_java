package ru.avicomp;

import java.io.*;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.compress.utils.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Useful utils to prepare test data.
 * <p>
 * Created by @szuev on 08.11.2017.
 */
@SuppressWarnings("WeakerAccess")
public class ShellUtils {
    private static final Logger LOGGER = LoggerFactory.getLogger(ShellUtils.class);
    private static final Charset DEFAULT_CHARSET = StandardCharsets.UTF_8;

    /**
     * Downloads a web-resource to local fs.
     *
     * @param src {@link URL}, the source
     * @param dst {@link Path}, the target
     * @throws IOException if something is wrong
     */
    public static void download(URL src, Path dst) throws IOException {
        LOGGER.info("WGET {} => {}", src, dst);
        try (ReadableByteChannel rbc = Channels.newChannel(src.openStream());
             FileOutputStream fos = new FileOutputStream(dst.toFile())) {
            fos.getChannel().transferFrom(rbc, 0, Long.MAX_VALUE);
        }
        LOGGER.debug("Finish downloading.");
    }

    /**
     * Unpacks archive *tar.gz
     *
     * @param src {@link Path}, the source
     * @param dst {@link Path}, the target
     * @throws IOException if something is wrong
     */
    public static void unpackTarGZ(Path src, Path dst) throws IOException {
        try (InputStream fis = Files.newInputStream(src);
             BufferedInputStream bin = new BufferedInputStream(fis);
             GzipCompressorInputStream gzIn = new GzipCompressorInputStream(bin);
             TarArchiveInputStream tarIn = new TarArchiveInputStream(gzIn)) {
            TarArchiveEntry ae;
            while ((ae = (TarArchiveEntry) tarIn.getNextEntry()) != null) {
                Path name = dst.resolve(ae.getName());
                if (ae.isDirectory()) {
                    LOGGER.debug("Make dir {}", name);
                    Files.createDirectories(name);
                } else {
                    LOGGER.debug("Unpack file {}", name);
                    try (OutputStream out = Files.newOutputStream(name)) {
                        IOUtils.copy(tarIn, out);
                    }
                }
            }
        }
    }

    /**
     * From classification-example.sh
     *
     * @param src {@link Path}, the source
     * @param dst {@link Path}, the target
     * @throws IOException if something is wrong
     */
    public static void normalize(Path src, Path dst) throws IOException {
        LOGGER.debug("Normalize file {}(=>{})", src, dst);
        transform(src, dst, ShellUtils::transform);
    }

    /**
     * Shuffles the content of file to new file. The source should be small to fit in memory.
     *
     * @param src {@link Path}, the source
     * @param dst {@link Path}, the target
     * @throws IOException if something is wrong
     */
    public static void shuffle(Path src, Path dst) throws IOException {
        LOGGER.debug("Shuffle file {}(<={})", dst, src);
        List<String> all = Files.lines(src, DEFAULT_CHARSET).collect(Collectors.toList());
        Collections.shuffle(all);
        Files.write(dst, all, DEFAULT_CHARSET);
    }

    /**
     * Normalizes and shuffles text file through creation of temporary file
     *
     * @param src {@link Path}, the source
     * @param dst {@link Path}, the target
     * @throws IOException if something is wrong
     */
    public static void normalizeAndShuffle(Path src, Path dst) throws IOException {
        Path tmp = Files.createTempFile(src.getFileName().toString() + "-" + dst.getFileName().toString() + "-", ".tmp");
        normalize(src, tmp);
        shuffle(tmp, dst);
        Files.delete(tmp);
    }

    /**
     * Transforms all rows from source file and writes them to destination.
     *
     * @param src {@link Path}, the source
     * @param dst {@link Path}, the target
     * @param map the {@link Function} to map line
     * @throws IOException if something is wrong
     */
    public static void transform(Path src, Path dst, Function<String, String> map) throws IOException {
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(dst, DEFAULT_CHARSET));
             Stream<String> lines = Files.lines(src, DEFAULT_CHARSET)) {
            lines.map(map).forEach(out::println);
        }
    }

    /**
     * The command:
     * <pre>{@code
     *   tr '[:upper:]' '[:lower:]' | sed -e 's/^/__label__/g' | \
     * sed -e "s/'/ ' /g" -e 's/"//g' -e 's/\./ \. /g' -e 's/<br \/>/ /g' \
     * -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' \
     * -e 's/\?/ \? /g' -e 's/\;/ /g' -e 's/\:/ /g' | tr -s " " | myshuf
     * }</pre>
     *
     * @param str String
     * @return String
     */
    public static String transform(String str) {
        return "__label__" + toLowerCase(str)
                .replaceAll("([,.'()!?])", " $1 ")
                .replace("\"", "")
                .replace("<br \\/>", " ")
                .replace(";", " ")
                .replace(":", " ")
                .replaceAll("[ ]+", " "); // tr -s " "
    }

    /**
     * The analogue of "tr '[:upper:]' '[:lower:]'"
     *
     * @param str String
     * @return String
     */
    public static String toLowerCase(String str) {
        StringBuilder sb = new StringBuilder();
        for (char c : str.toCharArray()) {
            if (c <= 127) {
                c = Character.toLowerCase(c);
            }
            sb.append(c);
        }
        return sb.toString();
    }
}
