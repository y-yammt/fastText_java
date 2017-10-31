package ru.avicomp;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by @szuev on 24.10.2017.
 */
public class TestsBase {
    public static final Logger LOGGER = LoggerFactory.getLogger("TESTS");
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

    public static String[] cmd(String cmd, Object in, Object out) {
        String res = String.format(cmd, in, out);
        LOGGER.info("Cmd: {}", res);
        return res.split("\\s");
    }

    public static String[] cmd(Data data) throws IOException, URISyntaxException {
        return cmd(data.cmd(), data.getInput(), data.getOutput());
    }

    public enum Data {
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
                return "junit.dbpedia.suprervised.t4.d10.lr01.wn2.b1e7.e5.m1";
            }
        };

        public abstract String input();

        public abstract String cmd();

        public abstract long binSize();

        public abstract long vecSize();

        public abstract int vecDim();

        public abstract int vecWords();

        public abstract String model();

        public Path getInput() throws URISyntaxException, IOException {
            return Paths.get(Data.class.getResource(input()).toURI()).toRealPath();
        }

        public Path getOutput() {
            return DESTINATION_DIR.resolve(model());
        }

        public Path getModelBin() {
            return Paths.get(getOutput().toString() + ".bin");
        }

        public Path getModelVec() {
            return Paths.get(getOutput().toString() + ".vec");
        }
    }
}
