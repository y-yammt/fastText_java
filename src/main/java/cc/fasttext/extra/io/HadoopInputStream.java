package cc.fasttext.extra.io;

import java.io.IOException;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import ru.avicomp.io.ScrollableInputStream;

/**
 * Created by @szuev on 30.10.2017.
 */
public class HadoopInputStream extends ScrollableInputStream {
    private FSDataInputStream in;
    private long size;

    public HadoopInputStream(FileSystem fs, Path path) throws IOException {
        this.size = fs.getFileStatus(path).getLen();
        this.in = fs.open(path);
    }

    @Override
    public void seek(long pos) throws IOException {
        in.seek(pos);
    }

    @Override
    public long getPos() throws IOException {
        return in.getPos();
    }

    @Override
    public long getLen() throws IOException {
        return size;
    }

    @Override
    public int read() throws IOException {
        return in.read();
    }

    @Override
    public int read(byte[] b) throws IOException {
        return in.read(b);
    }

    @Override
    public int read(byte[] b, int off, int len) throws IOException {
        return in.read(b, off, len);
    }

    @Override
    public long skip(long n) throws IOException {
        return in.skip(n);
    }

    @Override
    public int available() throws IOException {
        return in.available();
    }

    @Override
    public void close() throws IOException {
        in.close();
    }

    @Override
    public void mark(int limit) {
        in.mark(limit);
    }

    @Override
    public void reset() throws IOException {
        in.reset();
    }

    @Override
    public boolean markSupported() {
        return in.markSupported();
    }
}
