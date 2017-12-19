package cc.fasttext.extra.io;

import cc.fasttext.io.IOStreams;
import cc.fasttext.io.ScrollableInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;


/**
 * Created by @szuev on 30.10.2017.
 */
public class HadoopIOStreams implements IOStreams {
    private final FileSystem fs;

    public HadoopIOStreams(FileSystem fs) {
        this.fs = fs;
    }

    @Override
    public OutputStream createOutput(String uri) throws IOException {
        return fs.create(toPath(uri));
    }

    @Override
    public InputStream openInput(String uri) throws IOException {
        return fs.open(toPath(uri));
    }

    @Override
    public ScrollableInputStream openScrollable(String uri) throws IOException {
        return new HadoopInputStream(fs, toPath(uri));
    }

    @Override
    public long size(String uri) throws IOException {
        return fs.getFileStatus(toPath(uri)).getLen();
    }

    /**
     * Makes a hadoop fs path
     *
     * @param uri String
     * @return {@link Path hdfs path}
     */
    public static Path toPath(String uri) {
        return new Path(IOStreams.toURI(uri).getPath());
    }
}
