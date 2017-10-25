package ru.avicomp.io;

import java.io.IOException;
import java.io.InputStream;

/**
 * The {@link InputStream input stream} provider
 * <p>
 * Created by @szuev on 23.10.2017.
 */
@FunctionalInterface
public interface InputStreamSupplier {

    InputStream open() throws IOException;

    default long bytes() throws IOException, UnsupportedOperationException {
        throw new UnsupportedOperationException();
    }
}
