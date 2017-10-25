package ru.avicomp.io;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * Factory to create {@link java.io.InputStream} and {@link java.io.OutputStream} with the same nature, depending on file System.
 * <p>
 * Created by @szuev on 24.10.2017.
 */
public interface IOStreams {

    InputStreamSupplier createInput(String path);

    OutputStreamSupplier createOutput(String path);

    default OutputStream create(String path) throws IOException {
        return createOutput(path).open();
    }

    default InputStream open(String path) throws IOException {
        return createInput(path).open();
    }

    default boolean canRead(String path) {
        return true;
    }

    default boolean canWrite(String path) {
        return true;
    }

    default void prepare(String path) throws IOException {

    }
}
