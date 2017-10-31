package ru.avicomp;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import cc.fasttext.Main;

/**
 * Created by @szuev on 20.10.2017.
 */
@RunWith(Parameterized.class)
public class TrainModelTest {

    private final TestsBase.Data data;

    public TrainModelTest(TestsBase.Data data) {
        this.data = data;
    }

    @Parameterized.Parameters(name = "{0}")
    public static List<TestsBase.Data> getData() {
        return Arrays.asList(TestsBase.Data.values());
    }

    @Test
    public void test() throws Exception {
        new Main().train(TestsBase.cmd(data));

        Path bin = data.getModelBin();
        Path vec = data.getModelVec();
        Assert.assertTrue("No .bin", Files.exists(bin));
        Assert.assertTrue("No .vec", Files.exists(vec));

        // validate bin:
        long actualBinSize = Files.size(bin);
        Assert.assertEquals("Incorrect bin size: " + actualBinSize, data.binSize(), actualBinSize);

        // validate vec:
        int allowableDiffInPercents = 10;
        long actualVecSize = Files.size(vec);
        double actualDiffInPercents = 200.0 * (actualVecSize - data.vecSize()) / (actualVecSize + data.vecSize());
        TestsBase.LOGGER.info(String.format("Actual vec diff: %.2f%% (size: %d)", actualDiffInPercents, actualVecSize));
        Assert.assertTrue("Incorrect vec size: " + actualVecSize + ", diff: " + actualDiffInPercents, Math.abs(actualDiffInPercents) <= allowableDiffInPercents);
        List<Word> words = collect(vec);
        TestsBase.LOGGER.info("{}", toSet(words));
        Assert.assertEquals("Wrong size", data.vecWords(), words.size());
        Assert.assertTrue("Wrong dim inside file", toMap(words).values().stream().allMatch(floats -> floats.size() == data.vecDim()));
        try (BufferedReader r = Files.newBufferedReader(vec)) {
            Assert.assertEquals("Wrong first line", data.vecWords() + " " + data.vecDim(), r.lines().findFirst().orElseThrow(AssertionError::new));
        }
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
