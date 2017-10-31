package ru.avicomp.hdfs.io;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import ru.avicomp.io.IOStreams;
import ru.avicomp.io.ScrollableInputStream;

/**
 * Created by @szuev on 30.10.2017.
 */
public class HadoopIOStreams implements IOStreams {
    private final FileSystem fs;

    public HadoopIOStreams(FileSystem fs) {
        this.fs = fs;
    }

    @Override
    public OutputStream createOutput(String path) throws IOException {
        return fs.create(new Path(path));
    }

    @Override
    public InputStream openInput(String path) throws IOException {
        return fs.open(new Path(path));
    }

    @Override
    public ScrollableInputStream openScrollable(String path) throws IOException {
        return new HadoopInputStream(fs, new Path(path));
    }

    @Override
    public long size(String path) throws IOException {
        return fs.getFileStatus(new Path(path)).getLen();
    }
}
