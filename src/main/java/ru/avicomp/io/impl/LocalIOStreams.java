package ru.avicomp.io.impl;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import ru.avicomp.io.IOStreams;
import ru.avicomp.io.ScrollableInputStream;

/**
 * Created by @szuev on 30.10.2017.
 */
public class LocalIOStreams implements IOStreams {

    @Override
    public OutputStream createOutput(String file) throws IOException {
        return Files.newOutputStream(Paths.get(file));
    }

    @Override
    public InputStream openInput(String path) throws IOException {
        return Files.newInputStream(Paths.get(path));
    }

    @Override
    public boolean canRead(String path) {
        Path file = Paths.get(path);
        return Files.isRegularFile(file) && Files.isReadable(file);
    }

    @Override
    public boolean canWrite(String path) {
        Path parent = getParent(path);
        return parent != null && Files.isWritable(parent);
    }

    private Path getParent(String path) {
        Path file = Paths.get(path);
        Path res = file.getParent();
        if (res == null) {
            res = file.toAbsolutePath().getParent();
        }
        return res;
    }

    @Override
    public void prepareParent(String path) throws IOException {
        Path parent = getParent(path);
        if (parent == null) throw new IOException("No parent for " + path);
        Files.createDirectories(parent);
        Files.deleteIfExists(parent.resolve(path));
    }

    @Override
    public ScrollableInputStream openScrollable(String path) throws IOException {
        return new LocalInputStream(Paths.get(path));
    }

    @Override
    public long size(String path) throws IOException {
        return Files.size(Paths.get(path));
    }

}
