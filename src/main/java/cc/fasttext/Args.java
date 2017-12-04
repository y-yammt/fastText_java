package cc.fasttext;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Objects;
import java.util.function.IntFunction;

import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;

import ru.avicomp.io.FTInputStream;
import ru.avicomp.io.FTOutputStream;

/**
 * Immutable Args object.
 * See:
 * <a href='https://github.com/facebookresearch/fastText/blob/master/src/args.cc'>args.cc</a> and
 * <a href='https://github.com/facebookresearch/fastText/blob/master/src/args.h'>args.h</a>
 * TODO: retrain, input, output - must be excluded, charset, random factory - must be moved to fastText
 */
public final strictfp class Args {
    public static final String DEFAULT_LABEL = "__label__";
    // basic:
    private ModelName model = ModelName.SG;
    // todo: input & output should not be in args
    public String input;
    public String output;
    private int verbose = 2;
    // dictionary:
    private int minCount = 5;
    private int minCountLabel = 0;
    private int wordNgrams = 1;
    private int bucket = 2_000_000;
    private int minn = 3;
    private int maxn = 6;
    private double t = 1e-4;
    private String label = DEFAULT_LABEL;
    // training:
    private double lr = 0.05;
    private int lrUpdateRate = 100;
    private int dim = 100;
    private int ws = 5;
    private int epoch = 5;
    private int neg = 5;
    private LossName loss = LossName.NS;
    private int thread = 1;
    private String pretrainedVectors = "";
    private int saveOutput;
    // quantization:
    private boolean qout;
    private boolean retrain;
    private boolean qnorm;
    private int dsub = 2;
    private int cutoff;
    // additional:
    private Charset charset = StandardCharsets.UTF_8;
    private IntFunction<RandomGenerator> randomFactory = JDKRandomGenerator::new;

    public IntFunction<RandomGenerator> randomFactory() {
        return randomFactory;
    }

    public Charset charset() {
        return charset;
    }

    public ModelName model() {
        return model;
    }

    public LossName loss() {
        return loss;
    }

    public double lr() {
        return lr;
    }

    public int lrUpdateRate() {
        return lrUpdateRate;
    }

    public int dim() {
        return dim;
    }

    public int ws() {
        return ws;
    }

    public int epoch() {
        return epoch;
    }

    public int neg() {
        return neg;
    }

    public int wordNgrams() {
        return wordNgrams;
    }

    public int bucket() {
        return bucket;
    }

    public int minn() {
        return minn;
    }

    public int maxn() {
        return maxn;
    }

    public int minCount() {
        return minCount;
    }

    public int minCountLabel() {
        return minCountLabel;
    }

    public int thread() {
        return thread;
    }

    public double samplingThreshold() {
        return t;
    }

    public String label() {
        return label;
    }

    public String pretrainedVectors() {
        return pretrainedVectors;
    }

    public int verbose() {
        return verbose;
    }

    public boolean qout() {
        return qout;
    }

    public boolean retrain() {
        return retrain;
    }

    public boolean qnorm() {
        return qnorm;
    }

    public int dsub() {
        return dsub;
    }

    public int cutoff() {
        return cutoff;
    }

    public int saveOutput() {
        return saveOutput;
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
     * @return new instance of Args
     * @throws IOException if an I/O error occurs
     */
    static Args load(FTInputStream in) throws IOException {
        return new Builder()
                .setDim(in.readInt())
                .setWS(in.readInt())
                .setEpoch(in.readInt())
                .setMinCount(in.readInt())
                .setNeg(in.readInt())
                .setWordNgrams(in.readInt())
                .setLossName(LossName.fromValue(in.readInt()))
                .setModel(ModelName.fromValue(in.readInt()))
                .setBucket(in.readInt())
                .setMinN(in.readInt())
                .setMaxN(in.readInt())
                .setLRUpdateRate(in.readInt())
                .setSamplingThreshold(in.readDouble())
                .build();
    }

    /**
     * The Class-Builder to make new {@link Args args} object.
     * Must be the only way to achieve new instance of {@link Args args}.
     */
    public static class Builder {
        private Args _args = new Args();

        public Builder copy(Args other) {
            return setCharset(other.charset()).setRandomFactory(other.randomFactory())
                    .setModel(other.model()).setLossName(other.loss())
                    .setDim(other.dim()).setWS(other.ws()).setLR(other.lr()).setLRUpdateRate(other.lrUpdateRate()).setWordNgrams(other.wordNgrams())
                    .setMinCount(other.minCount()).setMinCountLabel(other.minCountLabel())
                    .setNeg(other.neg()).setBucket(other.bucket()).setMinN(other.minn()).setMaxN(other.maxn())
                    .setEpoch(other.epoch()).setThread(other.thread()).setSamplingThreshold(other.samplingThreshold())
                    .setLabel(other.label()).setVerbose(other.verbose()).setPreparedVectors(other.pretrainedVectors())
                    .setSaveOutput(other.saveOutput())
                    .setQNorm(other.qnorm()).setRetrain(other.retrain()).setQOut(other.qout()).setCutOff(other.cutoff()).setDSub(other.dsub());
        }

        public Builder setCharset(Charset charset) {
            _args.charset = Objects.requireNonNull(charset, "Null charset");
            return this;
        }

        public Builder setRandomFactory(IntFunction<RandomGenerator> f) {
            _args.randomFactory = Objects.requireNonNull(f, "Null random factory");
            return this;
        }

        public Builder setModel(ModelName name) {
            _args.model = Objects.requireNonNull(name, "Null model name");
            return this;
        }

        public Builder setLossName(LossName name) {
            _args.loss = Objects.requireNonNull(name, "Null loss name");
            return this;
        }

        public Builder setDim(int dim) {
            _args.dim = dim;
            return this;
        }

        public Builder setWS(int ws) {
            _args.ws = ws;
            return this;
        }

        public Builder setLR(double lr) {
            _args.lr = lr;
            return this;
        }

        public Builder setLRUpdateRate(int lrUpdateRate) {
            _args.lrUpdateRate = lrUpdateRate;
            return this;
        }

        public Builder setWordNgrams(int wordNgrams) {
            _args.wordNgrams = wordNgrams;
            return this;
        }

        public Builder setMinCount(int minCount) {
            _args.minCount = minCount;
            return this;
        }

        public Builder setMinCountLabel(int minCountLabel) {
            _args.minCountLabel = minCountLabel;
            return this;
        }

        public Builder setNeg(int neg) {
            _args.neg = neg;
            return this;
        }

        public Builder setBucket(int bucket) {
            _args.bucket = bucket;
            return this;
        }

        public Builder setMinN(int minn) {
            _args.minn = minn;
            return this;
        }

        public Builder setMaxN(int maxn) {
            _args.maxn = maxn;
            return this;
        }

        public Builder setEpoch(int epoch) {
            _args.epoch = epoch;
            return this;
        }

        public Builder setThread(int thread) {
            _args.thread = thread;
            return this;
        }

        public Builder setSamplingThreshold(double t) {
            _args.t = t;
            return this;
        }

        public Builder setLabel(String label) {
            _args.label = Objects.requireNonNull(label, "Null label");
            return this;
        }

        public Builder setVerbose(int verbose) {
            _args.verbose = verbose;
            return this;
        }

        public Builder setPreparedVectors(String preparedVectors) {
            _args.pretrainedVectors = Objects.requireNonNull(preparedVectors, "Null pre-trained vectors");
            return this;
        }

        public Builder setSaveOutput(int saveOutput) {
            _args.saveOutput = saveOutput;
            return this;
        }

        public Builder setQNorm(boolean qnorm) {
            _args.qnorm = qnorm;
            return this;
        }

        public Builder setRetrain(boolean retrain) {
            _args.retrain = retrain;
            return this;
        }

        public Builder setQOut(boolean qout) {
            _args.qout = qout;
            return this;
        }

        public Builder setCutOff(int cutoff) {
            _args.cutoff = cutoff;
            return this;
        }

        public Builder setDSub(int dsub) {
            _args.dsub = dsub;
            return this;
        }

        public Args build() {
            if (ModelName.SUP.equals(_args.model)) {
                _args.loss = LossName.SOFTMAX;
                _args.minCount = 1;
                _args.minn = 0;
                _args.maxn = 0;
                _args.lr = 0.1;
            }
            if (_args.wordNgrams <= 1 && _args.maxn == 0) {
                _args.bucket = 0;
            }
            return _args;
        }
    }

    public enum ModelName {
        CBOW("cbow", 1), SG("skipgram", 2), SUP("supervised", 3);

        private final int value;
        private final String name;

        ModelName(String name, int value) {
            this.name = name;
            this.value = value;
        }

        public String getName() {
            return name;
        }

        public static ModelName fromValue(int value) throws IllegalArgumentException {
            return Arrays.stream(values()).filter(v -> v.value == value)
                    .findFirst().orElseThrow(() -> new IllegalArgumentException("Unknown model enum value: " + value));
        }

        public static ModelName fromName(String value) {
            return Arrays.stream(values()).filter(v -> v.name.equalsIgnoreCase(value))
                    .findFirst().orElseThrow(() -> new IllegalArgumentException("Unknown model name: " + value));
        }
    }

    public enum LossName {
        HS(1), NS(2), SOFTMAX(3);
        private final int value;

        LossName(int value) {
            this.value = value;
        }

        public static LossName fromValue(int value) throws IllegalArgumentException {
            return Arrays.stream(values()).filter(v -> v.value == value)
                    .findFirst().orElseThrow(() -> new IllegalArgumentException("Unknown loss enum value: " + value));
        }

        public static LossName fromName(String value) throws IllegalArgumentException {
            return Arrays.stream(values()).filter(v -> v.name().equalsIgnoreCase(value))
                    .findFirst().orElseThrow(() -> new IllegalArgumentException("Unknown loss name: " + value));
        }
    }

}
