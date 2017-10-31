package ru.avicomp.io;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * Factory to create {@link java.io.InputStream} and {@link java.io.OutputStream} with the same nature depending on file System.
 * <p>
 * Created by @szuev on 24.10.2017.
 */
public interface IOStreams {

    OutputStream createOutput(String path) throws IOException;

    InputStream openInput(String path) throws IOException;

    default boolean canRead(String path) {
        return true;
    }

    default boolean canWrite(String path) {
        return true;
    }

    /**
     * Prepares the file to write.
     * Usually for creating parent directories
     *
     * @param path String, path identifier to file entity.
     * @throws IOException if something goes wrong while preparation.
     */
    default void prepareParent(String path) throws IOException {

    }

    /**
     * @param path
     * @return
     * @throws IOException
     * @throws UnsupportedOperationException
     */
    default ScrollableInputStream openScrollable(String path) throws IOException, UnsupportedOperationException {
        // TODO: add default impl
        throw new UnsupportedOperationException();
    }

    /**
     * @param path
     * @return
     * @throws IOException
     * @throws UnsupportedOperationException
     */
    default long size(String path) throws IOException, UnsupportedOperationException {
        // TODO: add default impl
        throw new UnsupportedOperationException();
    }

}
