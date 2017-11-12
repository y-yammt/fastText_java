package cc.fasttext;

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Objects;
import java.util.function.IntFunction;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;

import ru.avicomp.io.FTInputStream;
import ru.avicomp.io.FTOutputStream;
import ru.avicomp.io.FTReader;
import ru.avicomp.io.IOStreams;
import ru.avicomp.io.impl.LocalIOStreams;

/**
 * TODO: make immutable
 * See:
 * <a href='https://github.com/facebookresearch/fastText/blob/master/src/args.cc'>args.cc</a> and
 * <a href='https://github.com/facebookresearch/fastText/blob/master/src/args.h'>args.h</a>
 */
public strictfp class Args {
    public static final String DEFAULT_LABEL = "__label__";
    public String input;
    public String output;
    public String test;
    public double lr = 0.05;
    public int dim = 100;
    public int ws = 5;
    public int epoch = 5;
    public int minCount = 5;
    public int minCountLabel = 0;
    public int neg = 5;
    public int wordNgrams = 1;
    public LossName loss = LossName.NS;
    public ModelName model = ModelName.SG;
    public int bucket = 2_000_000;
    public int minn = 3;
    public int maxn = 6;
    public int thread = 1; // todo: = 12
    public double t = 1e-4;
    public int lrUpdateRate = 100;
    public String label = DEFAULT_LABEL;
    public int verbose = 2;
    public String pretrainedVectors = "";
    // TODO:
    public int saveOutput;
    //TODO:
    public boolean qout;


    Charset charset = StandardCharsets.UTF_8;
    private IOStreams factory = new LocalIOStreams();

    //RandomGenerator res = new Well19937c();
    private IntFunction<RandomGenerator> randomFactory = JDKRandomGenerator::new;

    Args() {
    }

    public Args setInput(String input) {
        this.input = Objects.requireNonNull(input, "Null input");
        return this;
    }

    public Args setCharset(Charset charset) {
        this.charset = Objects.requireNonNull(charset, "Null charset");
        return this;
    }

    public Args setIOStreams(IOStreams factory) {
        this.factory = Objects.requireNonNull(factory, "Null factory");
        return this;
    }

    public IOStreams getIOStreams() {
        return factory;
    }

    public IntFunction<RandomGenerator> getRandomFactory() {
        return randomFactory;
    }

    public Charset getCharset() {
        return charset;
    }

    /**
     * TODO: move or delete
     *
     * @return
     * @throws IOException
     */
    public FTReader createReader() throws IOException { // todo: buff size is experimental
        return new FTReader(getIOStreams().openScrollable(input), charset, 100 * 1024);
    }

    /**
     * TODO: remove?
     * Creates a writer for the specified path to write character data.
     *
     * @param path {@link Path} the path to write
     * @return {@link FTOutputStream}
     * @throws IOException if something is wrong
     */
    public Writer createWriter(Path path) throws IOException {
        return new OutputStreamWriter(getIOStreams().createOutput(path.toString()), charset);
    }

    /**
     * TODO: remove?
     * Creates a FS OutputStream for specified path to write binary data in cpp style.
     *
     * @param path {@link Path} the path to write
     * @return {@link FTOutputStream}
     * @throws IOException if something is wrong
     */
    public FTOutputStream createOutputStream(Path path) throws IOException {
        return new FTOutputStream(getIOStreams().createOutput(path.toString()));
    }

    /**
     * TODO: remove?
     * Creates a FS InputStream for specified path to read binary data in cpp (little endian) style.
     *
     * @param path {@link Path} the path to read.
     * @return {@link FTOutputStream}
     * @throws IOException if something is wrong
     */
    public FTInputStream createInputStream(Path path) throws IOException {
        return new FTInputStream(getIOStreams().openInput(path.toString()));
    }

    public void printHelp() {
        System.out.println("\n" + "The following arguments are mandatory:\n"
                + "  -input              training file path\n"
                + "  -output             output file path\n\n"
                + "The following arguments are optional:\n"
                + "  -lr                 learning rate [" + lr + "]\n"
                + "  -lrUpdateRate       change the rate of updates for the learning rate [" + lrUpdateRate + "]\n"
                + "  -dim                size of word vectors [" + dim + "]\n"
                + "  -ws                 size of the context window [" + ws + "]\n"
                + "  -epoch              number of epochs [" + epoch + "]\n"
                + "  -minCount           minimal number of word occurences [" + minCount + "]\n"
                + "  -minCountLabel      minimal number of label occurences [" + minCountLabel + "]\n"
                + "  -neg                number of negatives sampled [" + neg + "]\n"
                + "  -wordNgrams         max length of word ngram [" + wordNgrams + "]\n"
                + "  -loss               loss function {ns, hs, softmax} [ns]\n"
                + "  -bucket             number of buckets [" + bucket + "]\n"
                + "  -minn               min length of char ngram [" + minn + "]\n"
                + "  -maxn               max length of char ngram [" + maxn + "]\n"
                + "  -thread             number of threads [" + thread + "]\n"
                + "  -t                  sampling threshold [" + t + "]\n"
                + "  -label              labels prefix [" + label + "]\n"
                + "  -verbose            verbosity level [" + verbose + "]\n"
                + "  -pretrainedVectors  pretrained word vectors for supervised learning []");
    }

    /**
     * <pre>{@code
     * void Args::save(std::ostream& out) {
     *  out.write((char*) &(dim), sizeof(int));
     *  out.write((char*) &(ws), sizeof(int));
     *  out.write((char*) &(epoch), sizeof(int));
     *  out.write((char*) &(minCount), sizeof(int));
     *  out.write((char*) &(neg), sizeof(int));
     *  out.write((char*) &(wordNgrams), sizeof(int));
     *  out.write((char*) &(loss), sizeof(loss_name));
     *  out.write((char*) &(model), sizeof(model_name));
     *  out.write((char*) &(bucket), sizeof(int));
     *  out.write((char*) &(minn), sizeof(int));
     *  out.write((char*) &(maxn), sizeof(int));
     *  out.write((char*) &(lrUpdateRate), sizeof(int));
     *  out.write((char*) &(t), sizeof(double));
     * }}</pre>
     *
     * @param out {@link FTOutputStream}
     * @throws IOException if an I/O error occurs
     */
    void save(FTOutputStream out) throws IOException {
        out.writeInt(dim);
        out.writeInt(ws);
        out.writeInt(epoch);
        out.writeInt(minCount);
        out.writeInt(neg);
        out.writeInt(wordNgrams);
        out.writeInt(loss.value);
        out.writeInt(model.value);
        out.writeInt(bucket);
        out.writeInt(minn);
        out.writeInt(maxn);
        out.writeInt(lrUpdateRate);
        out.writeDouble(t);
    }

    /**
     * <pre>{@code void Args::load(std::istream& in) {
     *  in.read((char*) &(dim), sizeof(int));
     *  in.read((char*) &(ws), sizeof(int));
     *  in.read((char*) &(epoch), sizeof(int));
     *  in.read((char*) &(minCount), sizeof(int));
     *  in.read((char*) &(neg), sizeof(int));
     *  in.read((char*) &(wordNgrams), sizeof(int));
     *  in.read((char*) &(loss), sizeof(loss_name));
     *  in.read((char*) &(model), sizeof(model_name));
     *  in.read((char*) &(bucket), sizeof(int));
     *  in.read((char*) &(minn), sizeof(int));
     *  in.read((char*) &(maxn), sizeof(int));
     *  in.read((char*) &(lrUpdateRate), sizeof(int));
     *  in.read((char*) &(t), sizeof(double));
     * }}</pre>
     *
     * @param in {@link FTInputStream}
     * @throws IOException if an I/O error occurs
     */
    void load(FTInputStream in) throws IOException {
        dim = in.readInt();
        ws = in.readInt();
        epoch = in.readInt();
        minCount = in.readInt();
        neg = in.readInt();
        wordNgrams = in.readInt();
        loss = LossName.fromValue(in.readInt());
        model = ModelName.fromValue(in.readInt());
        bucket = in.readInt();
        minn = in.readInt();
        maxn = in.readInt();
        lrUpdateRate = in.readInt();
        t = in.readDouble();
    }

    public Args setModelName(ModelName name) {
        if (Objects.requireNonNull(name, "Null model name").equals(ModelName.SUP)) {
            this.loss = LossName.SOFTMAX;
            this.minCount = 1;
            this.minn = 0;
            this.maxn = 0;
            this.lr = 0.1;
        }
        this.model = name;
        return this;
    }

    public Args setLossName(LossName name) {
        Objects.requireNonNull(name, "Null loss name");
        this.loss = name;
        return this;
    }

    public Args setDim(int dim) {
        this.dim = dim;
        return this;
    }

    public Args setLR(double lr) {
        this.lr = lr;
        return this;
    }

    public Args setWordNgrams(int wordNgrams) {
        this.wordNgrams = wordNgrams;
        return this;
    }

    public Args setMinCount(int minCount) {
        this.minCount = minCount;
        return this;
    }

    public Args setBucket(int bucket) {
        this.bucket = bucket;
        return this;
    }

    public Args setEpoch(int epoch) {
        this.epoch = epoch;
        return this;
    }

    public Args setThread(int thread) {
        this.thread = thread;
        return this;
    }

    public void parseArgs(String[] args) {
        String command = args[0];
        if ("supervised".equalsIgnoreCase(command)) {
            model = ModelName.SUP;
            loss = LossName.SOFTMAX;
            minCount = 1;
            minn = 0;
            maxn = 0;
            lr = 0.1;
        } else if ("cbow".equalsIgnoreCase(command)) {
            model = ModelName.CBOW;
        }
        int ai = 1;
        while (ai < args.length) {
            if (args[ai].charAt(0) != '-') {
                System.out.println("Provided argument without a dash! Usage:");
                printHelp();
                System.exit(1);
            }
            if ("-h".equals(args[ai])) {
                System.out.println("Here is the help! Usage:");
                printHelp();
                System.exit(1);
            } else if ("-input".equals(args[ai])) {
                input = args[ai + 1];
            } else if ("-test".equals(args[ai])) {
                test = args[ai + 1];
            } else if ("-output".equals(args[ai])) {
                output = args[ai + 1];
            } else if ("-lr".equals(args[ai])) {
                lr = Double.parseDouble(args[ai + 1]);
            } else if ("-lrUpdateRate".equals(args[ai])) {
                lrUpdateRate = Integer.parseInt(args[ai + 1]);
            } else if ("-dim".equals(args[ai])) {
                dim = Integer.parseInt(args[ai + 1]);
            } else if ("-ws".equals(args[ai])) {
                ws = Integer.parseInt(args[ai + 1]);
            } else if ("-epoch".equals(args[ai])) {
                epoch = Integer.parseInt(args[ai + 1]);
            } else if ("-minCount".equals(args[ai])) {
                minCount = Integer.parseInt(args[ai + 1]);
            } else if ("-minCountLabel".equals(args[ai])) {
                minCountLabel = Integer.parseInt(args[ai + 1]);
            } else if ("-neg".equals(args[ai])) {
                neg = Integer.parseInt(args[ai + 1]);
            } else if ("-wordNgrams".equals(args[ai])) {
                wordNgrams = Integer.parseInt(args[ai + 1]);
            } else if ("-loss".equals(args[ai])) {
                if ("hs".equalsIgnoreCase(args[ai + 1])) {
                    loss = LossName.HS;
                } else if ("ns".equalsIgnoreCase(args[ai + 1])) {
                    loss = LossName.NS;
                } else if ("softmax".equalsIgnoreCase(args[ai + 1])) {
                    loss = LossName.SOFTMAX;
                } else {
                    System.out.println("Unknown loss: " + args[ai + 1]);
                    printHelp();
                    System.exit(1);
                }
            } else if ("-bucket".equals(args[ai])) {
                bucket = Integer.parseInt(args[ai + 1]);
            } else if ("-minn".equals(args[ai])) {
                minn = Integer.parseInt(args[ai + 1]);
            } else if ("-maxn".equals(args[ai])) {
                maxn = Integer.parseInt(args[ai + 1]);
            } else if ("-thread".equals(args[ai])) {
                thread = Integer.parseInt(args[ai + 1]);
            } else if ("-t".equals(args[ai])) {
                t = Double.parseDouble(args[ai + 1]);
            } else if ("-label".equals(args[ai])) {
                label = args[ai + 1];
            } else if ("-verbose".equals(args[ai])) {
                verbose = Integer.parseInt(args[ai + 1]);
            } else if ("-pretrainedVectors".equals(args[ai])) {
                pretrainedVectors = args[ai + 1];
            } else {
                System.out.println("Unknown argument: " + args[ai]);
                printHelp();
                System.exit(1);
            }
            ai += 2;
        }
        if (Utils.isEmpty(input) || Utils.isEmpty(output)) {
            System.out.println("Empty input or output path.");
            printHelp();
            System.exit(1);
        }
        if (wordNgrams <= 1 && maxn == 0) {
            bucket = 0;
        }
    }

    @Override
    public String toString() {
        return String.format("Args [input=%s, output=%s, test=%s, lr=%s, lrUpdateRate=%d, dim=%d, " +
                        "ws=%d, epoch=%d, minCount=%d, minCountLabel=%d, neg=%d, wordNgrams=%d, loss=%s, " +
                        "model=%s, bucket=%d, minn=%d, maxn=%d, thread=%d, t=%s, label=%s, verbose=%d, pretrainedVectors=%s]",
                input, output, test, lr, lrUpdateRate, dim,
                ws, epoch, minCount, minCountLabel, neg, wordNgrams, loss,
                model, bucket, minn, maxn, thread, t, label, verbose, pretrainedVectors);
    }

    public enum ModelName {
        CBOW(1), SG(2), SUP(3);

        private int value;

        ModelName(int value) {
            this.value = value;
        }

        public static ModelName fromValue(int value) throws IllegalArgumentException {
            return Arrays.stream(values()).filter(v -> v.value == value)
                    .findFirst().orElseThrow(() -> new IllegalArgumentException("Unknown model_name enum value: " + value));
        }
    }

    public enum LossName {
        HS(1), NS(2), SOFTMAX(3);
        private int value;

        LossName(int value) {
            this.value = value;
        }

        public static LossName fromValue(final int value) throws IllegalArgumentException {
            return Arrays.stream(values()).filter(v -> v.value == value)
                    .findFirst().orElseThrow(() -> new IllegalArgumentException("Unknown loss_name enum value: " + value));
        }
    }

}
