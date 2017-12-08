package ru.avicomp.io;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * Abstract factory to create {@link java.io.InputStream} and {@link java.io.OutputStream} with the same nature
 * depending on the encapsulated file system.
 * <p>
 * Created by @szuev on 24.10.2017.
 */
public interface IOStreams {

    /**
     * Opens or creates a file, returning an output stream that may be used to write bytes to the file.
     * The resulting stream will not be buffered. The stream will be safe for access by multiple concurrent threads.
     * Truncates file if it exists.
     * May create parents of the file if it is possible.
     *
     * @param uri, the reference to the file.
     * @return {@link OutputStream} dependent on encapsulated file system.
     * @throws IOException if it is not possible to create new file or read existing.
     */
    OutputStream createOutput(String uri) throws IOException;

    /**
     * Opens a file, returning an input stream to read from the file.
     * The stream will not be buffered, and is not required to support the {@link InputStream#mark mark} or {@link InputStream#reset reset} methods.
     * The stream will be safe for access by multiple concurrent threads.
     *
     * @param uri, the reference to the file
     * @return {@link InputStream} dependent on encapsulated file system.
     * @throws IOException if something wrong, e.g. no file found.
     */
    InputStream openInput(String uri) throws IOException;

    /**
     * Checks that the specified URI is good enough to be read.
     * May access the file system.
     *
     * @param uri, the file URI
     * @return true in case file can be read.
     */
    default boolean canRead(String uri) {
        return true;
    }

    /**
     * Checks the specified URI is good enough to write new file.
     * May access the file system.
     *
     * @param uri, the file URI
     * @return true if no errors expected when writing a file.
     */
    default boolean canWrite(String uri) {
        return true;
    }

    /**
     * @param uri
     * @return
     * @throws IOException
     * @throws UnsupportedOperationException
     */
    default ScrollableInputStream openScrollable(String uri) throws IOException, UnsupportedOperationException {
        // TODO: add default impl
        throw new UnsupportedOperationException("TODO: implement");
    }

    /**
     * @param uri
     * @return
     * @throws IOException
     * @throws UnsupportedOperationException
     */
    default long size(String uri) throws IOException, UnsupportedOperationException {
        // TODO: add default impl
        throw new UnsupportedOperationException("TODO: implement");
    }

}
