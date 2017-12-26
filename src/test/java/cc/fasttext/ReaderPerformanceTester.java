package cc.fasttext;

import cc.fasttext.base.Tests;
import cc.fasttext.io.WordReader;
import org.apache.commons.math3.distribution.UniformIntegerDistribution;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Stream;

/**
 * Not a test,
 * todo: will be removed.
 * Created by @szuev on 14.12.2017.
 */
public class ReaderPerformanceTester {

    public static void main(String... args) {
        Path p;
        /*try {
            p = Paths.get(TmpReaderPerformance.class.getResource("/text-data.txt").toURI()).toRealPath();
        } catch (IOException | URISyntaxException e) {
            throw new RuntimeException(e);
        }*/
        p = Tests.DESTINATION_DIR.resolve("dbpedia.test");
        System.out.println(p);

        int num = 50;
        Map<Integer, Function<Path, Set<String>>> map = new HashMap<>();
        Map<Integer, List<Float>> res = new HashMap<>();

        map.put(1, ReaderPerformanceTester::dictionaryRead);
        map.put(2, ReaderPerformanceTester::nioLinesRead);
        map.put(3, ReaderPerformanceTester::bufferReaderRead);
        map.put(4, _p -> newDictionaryRead(_p, 8 * 1024));
        map.put(5, _p -> newDictionaryRead(_p, 100 * 1024));
        map.put(6, _p -> newDictionaryRead(_p, 200 * 1024)); // faster ?
        map.put(7, _p -> newDictionaryRead(_p, 1024 * 1024));


        UniformIntegerDistribution u = new UniformIntegerDistribution(1, map.size());
        while (!isMapFull(res, num)) {
            Integer k = u.sample();
            if (valuesSize(res, k) == num) continue;
            float r = test(k, map.get(k), p);
            res.computeIfAbsent(k, _k -> new ArrayList<>()).add(r);
        }
        System.out.println();
        res.forEach((s, nums) -> System.out.println(s + ":\tnum=" + nums.size() + ",\taverage=" +
                nums.stream().mapToDouble(j -> j).average().orElse(Double.NaN)));

    }

    private static boolean isMapFull(Map<Integer, List<Float>> res, int size) {
        if (res.isEmpty()) return false;
        for (Integer k : res.keySet()) {
            if (valuesSize(res, k) < size) return false;
        }
        return true;
    }

    private static <K, V> int valuesSize(Map<K, List<V>> map, K k) {
        return !map.containsKey(k) ? 0 : map.get(k).size();
    }

    private static Set<String> dictionaryRead(Path p, Integer buffSize) {
        Set<String> res = new LinkedHashSet<>();
        try (Reader r = new BufferedReader(new InputStreamReader(Files.newInputStream(p), StandardCharsets.UTF_8), buffSize)) {
            String w;
            while ((w = DictionaryTest.readWord(r)) != null) {
                res.add(w);
            }
            return res;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static Set<String> dictionaryRead(Path p) {
        return dictionaryRead(p, 8 * 1024);
    }

    private static Set<String> newDictionaryRead(Path p, int buff) {
        Set<String> res = new LinkedHashSet<>();
        try (WordReader r = Dictionary.createWordReader(Files.newInputStream(p), StandardCharsets.UTF_8, buff)) {
            String w;
            while ((w = r.nextWord()) != null) {
                res.add(w);
            }
            return res;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static float test(Integer id, Function<Path, Set<String>> func, Path p) {
        System.out.print(id + ":\t");
        Instant s = Instant.now();
        System.out.print(func.apply(p).size());
        Instant e = Instant.now();
        float t = ChronoUnit.NANOS.between(s, e) / 1_000_000_000f;
        System.out.println(",\tt=" + t);
        return t;
    }

    private static Set<String> nioLinesRead(Path p) {
        Set<String> res = new LinkedHashSet<>();
        try (Stream<String> r = Files.lines(p)) {
            r.flatMap(ReaderPerformanceTester::asStream).forEach(res::add);
            return res;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static Set<String> bufferReaderRead(Path p) {
        Set<String> res = new LinkedHashSet<>();
        try (BufferedReader r = Files.newBufferedReader(p)) {
            String l;
            while ((l = r.readLine()) != null) {
                Collections.addAll(res, split(l));
            }
            return res;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private static Stream<String> asStream(String line) {
        return Arrays.stream(split(line));
    }

    private static String[] split(String line) { // c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == 0x000b || c == '\f' || c == '\0'
        return line.split("\\s|\\t|\\v|\\f\0");
    }

}
