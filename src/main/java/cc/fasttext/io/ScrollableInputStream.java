package cc.fasttext.io;

import java.io.IOException;
import java.io.InputStream;

/**
 * An abstract InputStream which permits seeking.
 * NOTE: the file should not be changed during operation with this stream
 * <p>
 * Created by @szuev on 30.10.2017.
 * @see <a href='https://github.com/kohsuke/hadoop/blob/master/src/core/org/apache/hadoop/fs/Seekable.java'>org.apache.hadoop.fs.Seekable</a>
 */
public abstract class ScrollableInputStream extends InputStream {

    /**
     * Seeks to the given offset in bytes from the start of the stream.
     * The next read() will be from that location.
     * Can't seek past the end of the file.
     *
     * @param bytes long
     * @throws IOException I/O error
     */
    public abstract void seek(long bytes) throws IOException;

    /**
     * Returns the current offset from the start of the file
     *
     * @return long
     * @throws IOException I/O error
     */
    public abstract long getPos() throws IOException;

    /**
     * Gets "length" of stream.
     * @return long
     * @throws IOException I/O error
     */
    public abstract long getLen() throws IOException;

    /**
     * Answers iff the end of stream is reached.
     * @return boolean
     * @throws IOException I/O error
     */
    public boolean isEnd() throws IOException {
        return getLen() == getPos();
    }

    /**
     * Returns {@code true} if current position is zero.
     * @return boolean
     * @throws IOException I/O error
     */
    public boolean isStart() throws IOException {
        return getPos() == 0;
    }
}
