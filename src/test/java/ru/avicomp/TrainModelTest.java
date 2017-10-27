package ru.avicomp;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;

import cc.fasttext.Args;
import cc.fasttext.FastText;
import cc.fasttext.Main;
import ru.avicomp.hdfs.HadoopMain;
import ru.avicomp.io.IOStreams;

/**
 * Created by @szuev on 20.10.2017.
 */
public class TrainModelTest {

    @Test
    public void cboxThread4Dim128Ws5Epoch10MinCount5test() throws Exception {
        Path input = Paths.get(TrainModelTest.class.getResource("/text-data.txt").toURI());
        Path output = TestsBase.DESTINATION_DIR.resolve("junit.test-data.cbox.d128.w5.hs");
        new Main().train(TestsBase.cmd("cbow -thread 4 -dim 128 -ws 5 -epoch 10 -minCount 5 -input %s -output %s", input, output));

        Path bin = Paths.get(output.toString() + ".bin");
        Path vec = Paths.get(output.toString() + ".vec");
        Assert.assertTrue("No .bin", Files.exists(bin));
        Assert.assertTrue("No .vec", Files.exists(vec));

        // validate bin:
        long expectedBinSize = 1_024_344_256;
        long actualBinSize = Files.size(bin);
        Assert.assertEquals("Incorrect bin size: " + actualBinSize, expectedBinSize, actualBinSize);

        // validate vec:
        int allowableDiffInPercents = 10;
        long expectedVecSize = 440_017;
        long actualVecSize = Files.size(vec);
        int actualDiffInPercents = (int) (200.0 * Math.abs(actualVecSize - expectedVecSize) / (actualVecSize + expectedVecSize));
        System.out.println("Actual diff: " + actualDiffInPercents + "%");
        Assert.assertTrue("Incorrect vec size: " + actualVecSize + ", diff: " + actualDiffInPercents, actualDiffInPercents <= allowableDiffInPercents);
        List<Word> words = collect(vec);
        System.out.println(toSet(words));
        int expectedDim = 128;
        int expectedSize = 331;
        Assert.assertEquals("Wrong size", expectedSize, words.size());
        Assert.assertTrue("Wrong dim inside file", toMap(words).values().stream().allMatch(floats -> floats.size() == expectedDim));
        try (BufferedReader r = Files.newBufferedReader(vec)) {
            Assert.assertEquals("Wrong first line", expectedSize + " " + expectedDim, r.lines().findFirst().orElseThrow(AssertionError::new));
        }
    }

    @Ignore // ignore since it requires preconfigured hadoop with test data inside fs
    @Test
    public void hadoopCboxThread4Dim128Ws5Epoch10MinCount5test() throws Exception {
        String hadoopHome = Paths.get(TrainModelTest.class.getResource("/bin").toURI()).getParent().toString();
        Map<String, String> props = new HashMap<>();
        props.put("hadoop.home.dir", hadoopHome);

        IOStreams fs = HadoopMain.createHadoopFS("hdfs://172.16.35.1:54310", "hadoop", Collections.emptyMap(), props);
        System.out.println(fs);
        String file = "raw-text-data.txt";
        String dir = "/tmp/out";
        String input = String.format("%s/%s", dir, file);
        String output = String.format("%s/junit.%s.d128.w5.hs", dir, file);

        String[] cmd = TestsBase.cmd("cbow -thread 4 -dim 128 -ws 5 -epoch 10 -minCount 5 -verbose 2 -input %s -output %s", input, output);
        Args args = Main.parseArgs(cmd).setIOStreams(fs);
        FastText fasttext = new FastText(args);
        fasttext.train();
    }

    private static class Word implements Comparable<Word> {
        String word;
        List<Float> vec;
        String in;

        private Word(String line) {
            String[] arr = (in = line).split("\\s");
            word = arr[0];
            vec = Arrays.stream(arr).skip(1).map(Float::parseFloat).collect(Collectors.toList());
        }

        @Override
        public String toString() {
            return String.format("%s:::%s}", word, vec);
        }

        @Override
        public int compareTo(Word o) {
            return word.compareTo(o.word);
        }
    }

    private static Set<String> toSet(List<Word> in) {
        return in.stream().map(w -> w.word).collect(Collectors.toSet());
    }

    private static Map<String, List<Float>> toMap(List<Word> in) {
        return in.stream().collect(Collectors.toMap(word -> word.word, word -> word.vec));
    }

    private static List<Word> collect(Path path) throws IOException {
        try (BufferedReader r = Files.newBufferedReader(path)) {
            return r.lines().skip(1).map(Word::new).collect(Collectors.toList());
        }
    }

}
