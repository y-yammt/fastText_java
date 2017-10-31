package ru.avicomp.io;

import java.io.*;
import java.nio.charset.Charset;
import java.util.Objects;

/**
 * FastText scrollable buffered input {@link Reader reader} with supporting mark/resets functionality.
 * <p>
 * Created by @szuev on 23.10.2017.
 */
@SuppressWarnings({"WeakerAccess", "UnusedReturnValue", "NullableProblems"})
public class FTReader extends Reader {
    private static final int DEFAULT_CHAR_BUFFER_LENGTH = 8192;
    private static final int UNKNOWN_POSITION = -2;

    private BufferedReader reader;
    private ScrollableInputStream stream;
    private long bytes;
    private int last;

    public FTReader(ScrollableInputStream stream, Charset charset) {
        this(stream, charset, DEFAULT_CHAR_BUFFER_LENGTH);
    }

    public FTReader(ScrollableInputStream stream, Charset charset, int bufferSize) {
        Objects.requireNonNull(charset, "Null encoding");
        if (bufferSize <= 0)
            throw new IllegalArgumentException("Buffer size <= 0");
        this.stream = Objects.requireNonNull(stream, "Null InputStream Supplier");
        this.reader = new BufferedReader(new InputStreamReader(stream, charset), bufferSize);
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
        try {
            reader.reset();
        } finally {
            last = UNKNOWN_POSITION;
        }
    }

    @Override
    public int read(char[] buff, int off, int len) throws IOException {
        return last = reader.read(Objects.requireNonNull(buff, "Null buf"), off, len);
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
        try {
            return reader.skip(n);
        } finally {
            last = UNKNOWN_POSITION;
        }
    }

    /**
     * @param n, long, the size of bytes to skip
     * @return long, actual size of skipped bytes
     * @throws IOException              if an I/O error occurs
     * @throws IllegalArgumentException if input is wrong
     * @see InputStream#skip(long)
     */
    public long skipBytes(long n) throws IOException {
        try {
            return stream.skip(n);
        } finally {
            last = UNKNOWN_POSITION;
        }
    }

    @Override
    public void close() throws IOException {
        try {
            reader.close();
        } finally {
            last = UNKNOWN_POSITION;
        }
    }

    /**
     * Resets stream to the initial position.
     *
     * @throws IOException if an I/O error occurs
     */
    public void rewind() throws IOException {
        try {
            stream.seek(0);
        } finally {
            last = UNKNOWN_POSITION;
        }
    }

    /**
     * Determines if it is the end of the character stream.
     *
     * @return true if end of stream is reached
     */
    public boolean end() throws IOException {
        if (last == -1) return true;
        if (stream.isEnd()) {
            last = -1;
            return true;
        }
        return false;
    }

    /**
     * Retrieves the size of source in bytes.
     *
     * @return long. the size of source in bytes.
     * @throws IOException some I/O error occurs
     */
    public long size() throws IOException {
        return bytes == 0 ? bytes = stream.getLen() : bytes;
    }
}
