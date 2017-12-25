package cc.fasttext;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Arrays;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.stream.Collectors;

import org.apache.commons.lang.StringUtils;

/**
 * Temporary class to measure time of events in runtime to gather statistics.
 * TODO: will be removed.
 * <p>
 * Created by @szuev on 25.12.2017.
 */
public enum Events {
    GET_FILE_SZIE,
    READ_DICT,
    IN_MATRIX_CREATE,
    OUT_MATRIX_CREATE,

    FILE_SEEK,
    DIC_GET_LINE,
    CBOW_CALC,
    DIC_GET_SUBWORDS_INT,
    MODEL_UPDATE,
    MODEL_COMPUTE_HIDDEN,
    MODEL_NEGATIVE_SAMPLING,
    MODEL_GRAD_MUL,
    MODEL_INPUT_ADD_ROW,
    CREATE_RES_MODEL,
    TRAIN,
    SAVE_BIN,;
    private ThreadLocal<Instant> start = new ThreadLocal<>();
    private ConcurrentLinkedQueue<Long> times = new ConcurrentLinkedQueue<>();

    public void start() {
        Instant now = Instant.now();
        if (start.get() != null) throw new IllegalStateException();
        start.set(now);
    }

    public void end() {
        Instant now = Instant.now();
        Instant start = this.start.get();
        if (start == null) throw new IllegalStateException();
        this.start.set(null);
        long time = ChronoUnit.MICROS.between(start, now);
        times.add(time);
    }

    public int size() {
        return times.size();
    }

    public double average() {
        return times.stream().mapToDouble(t -> t).average().orElse(Double.NaN) / 1_000_000;
    }

    public double sum() {
        return times.stream().mapToLong(t -> t).sum() / 1_000_000d;
    }

    public String toString() {
        return StringUtils.rightPad(name(), 40) +
                StringUtils.rightPad(String.valueOf(size()), 15) +
                StringUtils.rightPad(String.valueOf(average()), 25) +
                StringUtils.rightPad(String.valueOf(sum()), 20);
    }

    public static String print() {
        return Arrays.stream(values()).map(Events::toString).collect(Collectors.joining("\n"));
    }
}
