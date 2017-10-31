package ru.avicomp;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
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

    private final Data data;

    public TrainModelTest(Data data) {
        this.data = data;
    }

    @Parameterized.Parameters(name = "{0}")
    public static List<Data> getData() {
        return Arrays.asList(Data.values());
    }

    @Test
    public void test() throws Exception {
        Path input = Paths.get(TrainModelTest.class.getResource(data.input()).toURI());
        Path output = TestsBase.DESTINATION_DIR.resolve(data.model());
        new Main().train(TestsBase.cmd(data.cmd(), input, output));

        Path bin = Paths.get(output.toString() + ".bin");
        Path vec = Paths.get(output.toString() + ".vec");
        Assert.assertTrue("No .bin", Files.exists(bin));
        Assert.assertTrue("No .vec", Files.exists(vec));

        // validate bin:
        long actualBinSize = Files.size(bin);
        Assert.assertEquals("Incorrect bin size: " + actualBinSize, data.binSize(), actualBinSize);

        // validate vec:
        int allowableDiffInPercents = 10;
        long actualVecSize = Files.size(vec);
        double actualDiffInPercents = 200.0 * (actualVecSize - data.vecSize()) / (actualVecSize + data.vecSize());
        System.out.printf("Actual vec diff: %.2f%% (size: %d)%n", actualDiffInPercents, actualVecSize);
        Assert.assertTrue("Incorrect vec size: " + actualVecSize + ", diff: " + actualDiffInPercents, Math.abs(actualDiffInPercents) <= allowableDiffInPercents);
        List<Word> words = collect(vec);
        System.out.println(toSet(words));
        Assert.assertEquals("Wrong size", data.vecWords(), words.size());
        Assert.assertTrue("Wrong dim inside file", toMap(words).values().stream().allMatch(floats -> floats.size() == data.vecDim()));
        try (BufferedReader r = Files.newBufferedReader(vec)) {
            Assert.assertEquals("Wrong first line", data.vecWords() + " " + data.vecDim(), r.lines().findFirst().orElseThrow(AssertionError::new));
        }
    }

    private enum Data {
        CBOX_THREAD4_DIM128_WS5_EPOCH10_MINCOUNT5 {
            @Override
            public String input() {
                return "/text-data.txt";
            }

            @Override
            public String cmd() {
                return "cbow -thread 4 -dim 128 -ws 5 -epoch 10 -minCount 5 -input %s -output %s";
            }

            @Override
            public long binSize() {
                return 1_024_344_256;
            }

            @Override
            public long vecSize() {
                return 391_799;
            }

            @Override
            public int vecDim() {
                return 128;
            }

            @Override
            public int vecWords() {
                return 331;
            }

            public String model() {
                return "junit.text-data.cbox.t4.d128.w5.e10.m5";
            }
        },
        SUPERVISED_THREAD4_DIM10_LR01_NGRAMS2_BUCKET1E7_EPOCH5 {
            @Override
            public String input() {
                return "/dbpedia.cut.train";
            }

            @Override
            public String cmd() {
                return "supervised -dim 10 -lr 0.1 -wordNgrams 2 -minCount 1 -bucket 10000000 -epoch 5 -thread 4 -input %s -output %s";
            }

            @Override
            public long binSize() {
                return 400_154_095;
            }

            @Override
            public long vecSize() {
                return 279_763;
            }

            @Override
            public int vecDim() {
                return 10;
            }

            @Override
            public int vecWords() {
                return 2695;
            }

            @Override
            public String model() {
                return "junit.dbpedia.suprerbised.t4.d10.lr01.wn2.b1e7.e5.m1";
            }
        };

        public abstract String input();

        public abstract String cmd();

        public abstract long binSize();

        public abstract long vecSize();

        public abstract int vecDim();

        public abstract int vecWords();

        public abstract String model();
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
