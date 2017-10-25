package ru.avicomp.io;

import java.io.*;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.charset.Charset;
import java.util.Objects;
import java.util.Optional;

/**
 * FastText scrollable input {@link Reader reader} with supporting mark/resets functionality.
 * <p>
 * Created by @szuev on 23.10.2017.
 */
@SuppressWarnings({"WeakerAccess", "UnusedReturnValue"})
public class FSReader extends Reader {
    private static final int NOT_OPEN = -2;
    private static final int DEFAULT_CHAR_BUFFER_LENGTH = 8192;
    private static final int CALC_BYTES_SIZE_BUFFER_LENGTH = Integer.MAX_VALUE / 100;

    private final Charset charset;
    private final InputStreamSupplier source;
    private final int bufferSize;

    private BufferedReader reader;
    private InputStream stream;
    private int last = NOT_OPEN;
    private long sizeInBytes;

    public FSReader(InputStreamSupplier source, Charset charset) {
        this(source, charset, DEFAULT_CHAR_BUFFER_LENGTH);
    }

    public FSReader(InputStreamSupplier source, Charset charset, int bufferSize) {
        this.source = Objects.requireNonNull(source, "Null InputStream Supplier");
        this.charset = Objects.requireNonNull(charset, "Null encoding");
        if (bufferSize <= 0)
            throw new IllegalArgumentException("Buffer size <= 0");
        this.bufferSize = bufferSize;
    }

    @Override
    public boolean markSupported() {
        return true;
    }

    /**
     * @param limit, Limit on the number of characters that may be read while still preserving the mark.
     * @throws IOException              if an I/O error occurs
     * @throws NullPointerException     the source is not yet open
     * @throws IllegalArgumentException wrong limit
     * @see BufferedReader#mark(int)
     */
    @Override
    public void mark(int limit) throws IOException, NullPointerException {
        reader.mark(limit);
    }

    /**
     * @throws IOException          If the stream has never been marked, or if the mark has been invalidated.
     * @throws NullPointerException the source is not yet open
     * @see BufferedReader#reset()
     */
    public void reset() throws IOException, NullPointerException {
        reader.reset();
    }

    @Override
    public int read(char[] cbuf, int off, int len) throws IOException {
        return last = get().read(cbuf, off, len);
    }

    /**
     * @param n, long, the size of chars to skip
     * @return long, actual size of skipped chars
     * @throws IOException              if an I/O error occurs
     * @throws IllegalArgumentException if input is wrong
     * @see BufferedReader#skip(long)
     */
    @Override
    public long skip(long n) throws IOException {
        return skipChars(n);
    }

    /**
     * @param n, long, the size of chars to skip
     * @return long, actual size of skipped chars
     * @throws IOException              if an I/O error occurs
     * @throws IllegalArgumentException if input is wrong
     * @see Reader#skip(long)
     */
    public long skipChars(long n) throws IOException {
        return get().skip(n);
    }

    /**
     * @param n, long, the size of bytes to skip
     * @return long, actual size of skipped bytes
     * @throws IOException              if an I/O error occurs
     * @throws IllegalArgumentException if input is wrong
     * @see InputStream#skip(long)
     */
    public long skipBytes(long n) throws IOException {
        get();
        return stream.skip(n);
    }

    @Override
    public void close() throws IOException {
        try {
            if (reader == null) {
                return;
            }
            reader.close();
        } finally {
            clear();
        }
    }

    /**
     * Retrieves (creates ot fetches) new Reader
     *
     * @return {@link Reader}
     * @throws IOException if an I/O error occurs
     */
    protected Reader get() throws IOException {
        if (reader != null) return reader;
        return reader = new BufferedReader(new InputStreamReader(stream = source.open(), charset), bufferSize);
    }

    protected void clear() {
        reader = null;
        stream = null;
        last = NOT_OPEN;
    }

    public BufferedReader asBufferedReader() {
        return reader;
    }

    /**
     * Resets stream to the initial position.
     *
     * @throws IOException if an I/O error occurs
     */
    public void rewind() throws IOException {
        try {
            if (reader == null) {
                return;
            }
            Optional<FileChannel> channel = channel();
            if (channel.isPresent()) {
                channel.get().position(0);
                return;
            }
            close();
        } finally {
            last = NOT_OPEN;
        }
    }

    protected Optional<FileChannel> channel() {
        return Optional.ofNullable(stream)
                .map(Channels::newChannel)
                .filter(FileChannel.class::isInstance)
                .map(FileChannel.class::cast);
    }

    /**
     * Determines if it is the end of the character stream.
     *
     * @return true if end of stream is reached
     */
    public boolean end() {
        return last == -1;
    }

    /**
     * Retrieves the size of source in bytes.
     *
     * @return long. the size of source in bytes.
     * @throws IOException some I/O error occurs
     */
    public long size() throws IOException {
        return sizeInBytes == 0 ? sizeInBytes = calcSizeInBytes() : sizeInBytes;
    }

    protected long calcSizeInBytes() throws IOException {
        try {
            return source.bytes();
        } catch (UnsupportedOperationException e) {
            // ignore
        }
        if (reader != null) {
            Optional<FileChannel> channel = channel();
            if (channel.isPresent()) {
                return channel.get().size();
            }
        }
        try (InputStream in = source.open()) {
            long res;
            res = in.available();
            if (res != 0) return res;
            byte[] buff = new byte[CALC_BYTES_SIZE_BUFFER_LENGTH];
            int n;
            while ((n = in.read(buff)) != -1) {
                res += n;
            }
            return res;
        }
    }
}
