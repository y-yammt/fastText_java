package ru.avicomp.io;

import java.io.*;
import java.nio.charset.Charset;
import java.util.Objects;

/**
 * FastText scrollable buffered input {@link Reader reader} with supporting mark/resets functionality.
 * <p>
 * Created by @szuev on 23.10.2017.
 */
@SuppressWarnings({"WeakerAccess", "UnusedReturnValue", "NullableProblems", "SameParameterValue"})
public class FTReader extends Reader {
    private static final int DEFAULT_CHAR_BUFFER_LENGTH = 8192;
    private static final int UNKNOWN_POSITION = -2;

    private BufferedReader reader;
    private InputStream stream;
    private long bytes;
    private int last;

    public FTReader(InputStream stream, Charset charset) {
        this(stream, charset, DEFAULT_CHAR_BUFFER_LENGTH);
    }

    public FTReader(InputStream stream, Charset charset, int bufferSize) {
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
     * @throws IOException                   if an I/O error occurs
     * @throws UnsupportedOperationException if this operation is not supported on underling stream
     */
    public void rewind() throws IOException {
        try {
            seek(0);
        } finally {
            last = UNKNOWN_POSITION;
        }
    }

    protected void seek(int position) throws IOException, UnsupportedOperationException {
        if (stream instanceof ScrollableInputStream) {
            ((ScrollableInputStream) stream).seek(position);
            return;
        }
        throw new UnsupportedOperationException("Seek not supported");
    }

    /**
     * Determines if it is the end of the character stream.
     *
     * @return true if end of stream is reached
     */
    public boolean end() throws IOException {
        if (last == -1) return true;
        if (last == UNKNOWN_POSITION && isEnd()) {
            last = -1;
            return true;
        }
        return false;
    }

    protected boolean isEnd() throws IOException {
        if (stream instanceof ScrollableInputStream) {
            return ((ScrollableInputStream) stream).isEnd();
        }
        try {
            mark(1);
            return read() == -1;
        } finally {
            reset();
        }
    }

    /**
     * Retrieves the size of source in bytes.
     *
     * @return long. the size of source in bytes.
     * @throws IOException                   some I/O error occurs
     * @throws UnsupportedOperationException if this operation is not supported on underling stream
     */
    public long size() throws IOException, UnsupportedOperationException {
        return bytes == 0 ? bytes = calcSize() : bytes;
    }

    protected long calcSize() throws IOException, UnsupportedOperationException {
        if (stream instanceof ScrollableInputStream) {
            return ((ScrollableInputStream) stream).getLen();
        }
        throw new UnsupportedOperationException("Getting size is not supported");
    }
}
